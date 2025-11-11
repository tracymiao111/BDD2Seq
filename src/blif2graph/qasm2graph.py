#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qasm2graph_for_bdd2seq.py

Convert a QASM circuit into a DGL graph consistent with the bdd2seq/blif2graph style:
- Node types: 0=input, 1=gate(X/CX/CCX), 2=output
- Features: [index, type, topo_order, depth, fan_in, fan_out] + padded_truth_vector
- Optional: strip the initial H prefix and rebuild a complete H layer; detect and fold the 7-T CCX template.

Note: To ensure Boolean feature consistency, only X/CX/CCX gates are kept.
New: --mcx-policy {drop,error,truncate} (default: drop) for handling mcx with more than 3 arguments.
"""

import argparse, glob, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import networkx as nx
import dgl

# -----------------------------
# Lightweight QASM parsing (generic gate + argument list)
# -----------------------------

QREG_RE = re.compile(r'^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;', re.IGNORECASE)
CREG_RE = re.compile(r'^\s*creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;', re.IGNORECASE)
GENERIC_GATE_RE = re.compile(r'^\s*([a-z][a-z0-9_]*)\s+(.*?)\s*;\s*$', re.IGNORECASE)
QARG_RE = re.compile(r'^\s*(\w+)\s*\[\s*(\d+)\s*\]\s*$', re.IGNORECASE)

SKIP_RE_LIST = [
    re.compile(r'^\s*barrier\b', re.IGNORECASE),
    re.compile(r'^\s*measure\b', re.IGNORECASE),
    re.compile(r'^\s*//'),  # line comment
]

def read_qasm_lines(path: Path) -> List[str]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return [ln.strip() for ln in f]

def parse_qregs(lines: List[str]) -> Dict[str, int]:
    qregs: Dict[str, int] = {}
    for ln in lines:
        m = QREG_RE.match(ln)
        if m:
            name, n = m.group(1), int(m.group(2))
            qregs[name] = n
    if not qregs:
        raise ValueError("No qreg found.")
    return qregs

class Op:
    __slots__ = ('name', 'qargs')
    def __init__(self, name: str, qargs: List[Tuple[str,int]]):
        self.name = name.lower()
        self.qargs = qargs

    def __repr__(self): return f"Op({self.name}, {self.qargs})"

def parse_ops(lines: List[str]) -> List[Op]:
    ops: List[Op] = []
    for ln in lines:
        if not ln or any(pat.match(ln) for pat in SKIP_RE_LIST):
            continue
        m = GENERIC_GATE_RE.match(ln)
        if not m:
            continue
        gname, arglist = m.group(1).lower(), m.group(2)
        # support x/h/t/tdg/cx/ccx/mcx (others will be filtered later)
        parts = [p.strip() for p in arglist.split(',') if p.strip()]
        qargs: List[Tuple[str,int]] = []
        ok = True
        for p in parts:
            mq = QARG_RE.match(p)
            if not mq:
                ok = False
                break
            qargs.append((mq.group(1), int(mq.group(2))))
        if ok:
            ops.append(Op(gname, qargs))
    return ops

# -----------------------------
# Normalization: strip initial H layer & rebuild
# -----------------------------

def strip_initial_h_prefix(ops: List[Op]) -> Tuple[List[Op], Dict[Tuple[str,int], bool]]:
    """Remove consecutive initial h layer (possibly incomplete), return new ops and the stripped qubit set."""
    stripped = set()
    i = 0
    while i < len(ops) and ops[i].name == 'h' and len(ops[i].qargs) == 1:
        stripped.add(ops[i].qargs[0])
        i += 1
    return ops[i:], {q: True for q in stripped}

def prepend_full_h_layer(ops: List[Op], qregs: Dict[str,int]) -> List[Op]:
    """Add an H to every qubit line (ordered by qreg name and index)."""
    new = []
    for qr in sorted(qregs.keys()):
        for idx in range(qregs[qr]):
            new.append(Op('h', [(qr, idx)]))
    new.extend(ops)
    return new

# -----------------------------
# Fold 7-T CCX template → ccx macro
# -----------------------------

def _same_qubit(a: Tuple[str,int], b: Tuple[str,int]) -> bool:
    return a[0] == b[0] and a[1] == b[1]

def _match_seq(ops: List[Op], i: int) -> Optional[Tuple[int,Tuple[Tuple[str,int],Tuple[str,int],Tuple[str,int]]]]:
    """
    Try matching the 7-T template starting at position i:
        h c;
        cx b,c; tdg c; cx a,c; t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c;
        cx a,b; t a; tdg b; cx a,b;
    Return (end_index, (a,b,c)) if matched; None otherwise.
    """
    try:
        # h c
        if not (ops[i].name=='h' and len(ops[i].qargs)==1): return None
        c = ops[i].qargs[0]; k = i+1

        # cx b,c
        if not (ops[k].name=='cx' and _same_qubit(ops[k].qargs[1], c)): return None
        b = ops[k].qargs[0]; k+=1

        # tdg c
        if not (ops[k].name=='tdg' and _same_qubit(ops[k].qargs[0], c)): return None
        k+=1

        # cx a,c
        if not (ops[k].name=='cx' and _same_qubit(ops[k].qargs[1], c)): return None
        a = ops[k].qargs[0]; k+=1

        # t c
        if not (ops[k].name=='t' and _same_qubit(ops[k].qargs[0], c)): return None
        k+=1

        # cx b,c
        if not (ops[k].name=='cx' and _same_qubit(ops[k].qargs[1], c) and _same_qubit(ops[k].qargs[0], b)): return None
        k+=1

        # tdg c
        if not (ops[k].name=='tdg' and _same_qubit(ops[k].qargs[0], c)): return None
        k+=1

        # cx a,c
        if not (ops[k].name=='cx' and _same_qubit(ops[k].qargs[1], c) and _same_qubit(ops[k].qargs[0], a)): return None
        k+=1

        # t b
        if not (ops[k].name=='t' and _same_qubit(ops[k].qargs[0], b)): return None
        k+=1

        # t c
        if not (ops[k].name=='t' and _same_qubit(ops[k].qargs[0], c)): return None
        k+=1

        # h c
        if not (ops[k].name=='h' and _same_qubit(ops[k].qargs[0], c)): return None
        k+=1

        # cx a,b
        if not (ops[k].name=='cx' and _same_qubit(ops[k].qargs[0], a) and _same_qubit(ops[k].qargs[1], b)): return None
        k+=1

        # t a
        if not (ops[k].name=='t' and _same_qubit(ops[k].qargs[0], a)): return None
        k+=1

        # tdg b
        if not (ops[k].name=='tdg' and _same_qubit(ops[k].qargs[0], b)): return None
        k+=1

        # cx a,b
        if not (ops[k].name=='cx' and _same_qubit(ops[k].qargs[0], a) and _same_qubit(ops[k].qargs[1], b)): return None
        end = k

        return end, (a,b,c)
    except IndexError:
        return None

def fold_ct_ccx_templates(ops: List[Op]) -> List[Op]:
    """Fold standard 7-T templates into ccx macros (multiple passes, left-to-right)."""
    out: List[Op] = []
    i = 0
    while i < len(ops):
        m = _match_seq(ops, i)
        if m is None:
            out.append(ops[i]); i += 1
        else:
            end, (a,b,c) = m
            out.append(Op('ccx', [a,b,c]))
            i = end + 1
    return out

# -----------------------------
# Graph construction (consistent with blif2graph)
# -----------------------------

def tt_x() -> List[int]:          # t' = ~t
    return [1,0]

def tt_cx() -> List[int]:         # inputs [c,t], t' = t xor c
    return [0,1,1,0]

def tt_ccx() -> List[int]:        # inputs [a,b,t], t' = t xor (a&b)
    out = []
    for a in (0,1):
        for b in (0,1):
            for t in (0,1):
                out.append(t ^ (a & b))
    return out  # 8 entries

def gate_tt(name: str) -> List[int]:
    if name == 'x':   return tt_x()
    if name == 'cx':  return tt_cx()
    if name == 'ccx': return tt_ccx()
    raise ValueError(f"TT not defined for gate {name}")

def process_qasm_to_graph(
    qasm_path: Path,
    padding_length: int = 8,
    strip_and_rebuild_h: bool = True,
    fold_ccx_template: bool = True,
    mcx_policy: str = "drop",  # 'drop' | 'error' | 'truncate'
) -> Tuple[dgl.DGLGraph, str]:
    lines = read_qasm_lines(qasm_path)
    qregs = parse_qregs(lines)
    ops = parse_ops(lines)

    if strip_and_rebuild_h:
        ops, _ = strip_initial_h_prefix(ops)
        ops = prepend_full_h_layer(ops, qregs)

    if fold_ccx_template:
        ops = fold_ct_ccx_templates(ops)

    dropped_mcx, truncated_mcx = 0, 0
    filtered: List[Op] = []
    for op in ops:
        nqa = len(op.qargs)
        if op.name in ('x','h','t','tdg'):
            continue
        if op.name == 'cx':
            if nqa == 2:
                filtered.append(op)
            continue
        if op.name == 'ccx':
            if nqa == 3:
                filtered.append(op)
            continue
        if op.name == 'mcx':
            if nqa == 3:
                filtered.append(Op('ccx', op.qargs))
            elif nqa > 3:
                if mcx_policy == 'drop':
                    dropped_mcx += 1
                    continue
                elif mcx_policy == 'error':
                    raise RuntimeError(f"[mcx-policy=error] {qasm_path.name}: found mcx with {nqa} args.")
                elif mcx_policy == 'truncate':
                    truncated_mcx += 1
                    filtered.append(Op('ccx', [op.qargs[0], op.qargs[1], op.qargs[-1]]))
                else:
                    raise ValueError(f"Unknown mcx_policy: {mcx_policy}")
            continue
        continue

    G = nx.DiGraph()
    name2id: Dict[Tuple[str,int], int] = {}
    nid = 0

    for qr in sorted(qregs.keys()):
        for i in range(qregs[qr]):
            G.add_node(nid, name=f"{qr}[{i}]", type=0)
            name2id[(qr,i)] = nid
            nid += 1

    last_writer: Dict[Tuple[str,int], int] = dict(name2id)

    gate_node_gatekind: Dict[int, str] = {}

    for op in filtered:
        this_id = nid
        G.add_node(this_id, name=f"{op.name}", type=1)
        for q in op.qargs:
            G.add_edge(last_writer[q], this_id)
        tgt_idx = 0 if op.name == 'x' else (len(op.qargs) - 1)
        last_writer[op.qargs[tgt_idx]] = this_id
        gate_node_gatekind[this_id] = op.name
        nid += 1

    for qr in sorted(qregs.keys()):
        for i in range(qregs[qr]):
            out_id = nid
            G.add_node(out_id, name=f"{qr}[{i}]_out", type=2)
            G.add_edge(last_writer[(qr,i)], out_id)
            nid += 1

    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        raise RuntimeError(f"Graph from {qasm_path} is not a DAG.")
    topo_pos = {v:i for i,v in enumerate(topo)}

    depth = {v:0 for v in G.nodes}
    for v in topo:
        d = depth[v]
        for suc in G.successors(v):
            depth[suc] = max(depth[suc], d+1)

    fan_in  = dict(G.in_degree())
    fan_out = dict(G.out_degree())

    N = G.number_of_nodes()
    feature_vectors: List[List[float]] = [[0.0]* (6 + padding_length) for _ in range(N)]

    def pad_vec(v: List[int]) -> List[int]:
        if len(v) >= padding_length: return v[:padding_length]
        return v + [0]*(padding_length - len(v))

    for v in G.nodes:
        ntype = G.nodes[v]['type']
        base = [float(v), float(ntype), float(topo_pos[v]),
                float(depth[v]), float(fan_in[v]), float(fan_out[v])]
        if ntype != 1:
            tt = [0]*padding_length
        else:
            gk = gate_node_gatekind[v]
            tt = pad_vec(gate_tt(gk)) if gk in ('x','cx','ccx') else [0]*padding_length
        feature_vectors[v] = base + list(map(float, tt))

    edges = list(G.edges)
    src = [u for (u,_) in edges]
    dst = [w for (_,w) in edges]

    g = dgl.graph((src, dst), num_nodes=N)
    g.ndata['feat'] = torch.tensor(feature_vectors, dtype=torch.float)

    if dropped_mcx or truncated_mcx:
        print(f"[INFO] {qasm_path.name}: dropped_mcx={dropped_mcx}, truncated_mcx={truncated_mcx}")

    circuit_name = qasm_path.stem
    return g, circuit_name

# -----------------------------
# Batch processing & saving
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, help="QASM input root directory")
    ap.add_argument("--in-pattern", default="**/*.qasm",
                    help="Glob pattern relative to in-root (default **/*.qasm)")
    ap.add_argument("--padding", type=int, default=8, help="Truth table padding length (default 8)")
    ap.add_argument("--strip-rebuild-h", type=int, default=1,
                    help="Whether to strip prefix H and rebuild a complete H layer (1/0, default 1)")
    ap.add_argument("--fold-ct", type=int, default=1,
                    help="Whether to fold 7-T CCX templates (1/0, default 1)")
    ap.add_argument("--mcx-policy", choices=["drop","error","truncate"], default="drop",
                    help="Policy for handling mcx with >3 args (default drop)")
    ap.add_argument("--out", required=True, help="Output .pt path (torch.save dict)")
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    paths = [Path(p) for p in glob.glob(str(in_root / args.in_pattern), recursive=True)]
    paths = [p for p in paths if p.suffix.lower()==".qasm"]

    graphs, names = [], []
    for p in sorted(paths):
        try:
            g, name = process_qasm_to_graph(
                p, padding_length=args.padding,
                strip_and_rebuild_h=bool(args.strip_rebuild_h),
                fold_ccx_template=bool(args.fold_ct),
                mcx_policy=args.mcx_policy
            )
            graphs.append(g)
            names.append(name)
            print(f"[OK] {p.relative_to(in_root)} -> nodes={g.num_nodes()}, edges={g.num_edges()}")
        except Exception as e:
            print(f"[SKIP] {p}: {e}", file=sys.stderr)
            continue

    if not graphs:
        print("[ERR] No graphs generated.", file=sys.stderr)
        sys.exit(1)

    dataset