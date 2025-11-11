#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIG edgelist (*.el) -> DGLGraph dataset (fixed ordering)
- Node order: PIs (by port ascending) -> AIG gates (ascending id) -> POs (ascending id)
- Features per node: [index, type, topo, depth, fan_in, fan_out] + zeros(padding_length)
- Labels: read from {label_dir}/{circuit}_BDD_all/best_algorithm.txt
          must be a permutation of 0..num_pis-1 (convert 1-based -> 0-based if needed)
- Splits and saves to .pt files (train/val/test)
"""

import os
import re
import glob
import argparse
import random
from typing import Dict, List, Tuple, Optional, Set

import torch
import networkx as nx
import dgl


# ----------------------------
# I/O helpers
# ----------------------------
def read_variable_ordering(label_dir: str, circuit_name: str) -> List[int]:
    """
    Returns a list of integers from ".../{circuit}_BDD_all/best_algorithm.txt" line:
      Variable ordering: [a, b, c, ...]
    """
    label_file = os.path.join(label_dir, f"{circuit_name}_BDD_all", "best_algorithm.txt")
    if not os.path.exists(label_file):
        print(f"[WARNING] Label file not found for circuit '{circuit_name}' at {label_file}.")
        return []

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Variable ordering:"):
            ordering_str = line.strip().split(":", 1)[1].strip()
            ordering_str = ordering_str.strip("[]")
            if not ordering_str:
                print(f"[DEBUG] Circuit '{circuit_name}' has an empty variable ordering list.")
                return []
            try:
                ordering = [int(x.strip()) for x in ordering_str.split(",") if x.strip()]
            except ValueError:
                print(f"[ERROR] Circuit '{circuit_name}' ordering parse error: {ordering_str}")
                return []
            return ordering

    print(f"[DEBUG] Circuit '{circuit_name}' Variable Ordering line not found.")
    return []


def parse_el_file(filepath: str) -> Tuple[
    Dict[int, int],   # pi_port -> pi_node_id
    Set[int],         # pi_node_ids
    Set[int],         # gate_node_ids (dst of AIG)
    Set[int],         # po_node_ids (dst of Po)
    List[Tuple[int,int]]  # edges (src_id, dst_id) in original id space
]:
    """
    Parse ABC edgelist.
    Lines: "src dst TYPE flags"
      TYPE in {"Pi", "AIG", "Po"} (case-insensitive)
    """
    pi_port_to_node: Dict[int, int] = {}
    pi_nodes: Set[int] = set()
    gate_nodes: Set[int] = set()
    po_nodes: Set[int] = set()
    edges_raw: List[Tuple[int, int]] = []

    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                continue

            typ = parts[2].upper()
            # flags = parts[3]  # polarity bits; unused for features

            if typ == "PI":
                # 'a' is the external PI port index, 'b' is PI node id
                pi_port_to_node[a] = b
                pi_nodes.add(b)
            elif typ == "AIG":
                edges_raw.append((a, b))
                gate_nodes.add(b)
            elif typ == "PO":
                edges_raw.append((a, b))
                po_nodes.add(b)
            else:
                pass

    return pi_port_to_node, pi_nodes, gate_nodes, po_nodes, edges_raw


# ----------------------------
# Core builder
# ----------------------------
def build_graph_from_el(
    filepath: str,
    padding_length: int = 8,
    label_dir: Optional[str] = None,
):
    """
    Returns:
        g: DGLGraph with ndata['feat']
        labels: torch.LongTensor or None
        circuit_name: str
    """
    circuit_name = os.path.splitext(os.path.basename(filepath))[0]

    (pi_port_to_node,
     pi_nodes,
     gate_nodes,
     po_nodes,
     edges_raw) = parse_el_file(filepath)

    # --- Ensure we have at least one PI ---
    if not pi_port_to_node:
        print(f"[SKIPPED] '{circuit_name}' has no PIs.")
        return None, None, circuit_name

    # --- Build node order: PIs(by port asc) -> gates -> POs ---
    ports_sorted = sorted(pi_port_to_node.keys())  # 1-based ports typically
    input_nids_ordered = [pi_port_to_node[p] for p in ports_sorted]

    # gates only (exclude any PI/PO ids just in case)
    gate_only_ids = sorted(gate_nodes - pi_nodes - po_nodes)
    po_ids_ordered = sorted(po_nodes)

    ordered_nids = input_nids_ordered + gate_only_ids + po_ids_ordered
    num_nodes = len(ordered_nids)
    nid2idx = {nid: i for i, nid in enumerate(ordered_nids)}

    # --- Edges in new index space ---
    edges = []
    for u, v in edges_raw:
        if u in nid2idx and v in nid2idx:
            edges.append((nid2idx[u], nid2idx[v]))

    # --- Node types (0=PI, 1=Gate, 2=PO) ---
    node_types = [1] * num_nodes
    num_pis = len(input_nids_ordered)
    for i in range(num_pis):
        node_types[i] = 0
    start_gates = num_pis
    end_gates = num_pis + len(gate_only_ids)
    for i in range(start_gates, end_gates):
        node_types[i] = 1
    for i in range(end_gates, num_nodes):
        node_types[i] = 2

    # --- Build NX graph on new indices for topo/degree ---
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # DAG check & topo order
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print(f"[ERROR] Graph {circuit_name} is not a DAG. Skipping.")
        return None, None, circuit_name

    # Depth (longest path length) along topo
    depths = {i: 0 for i in range(num_nodes)}
    for u in topo_order:
        d = depths[u]
        for v in G.successors(u):
            if depths[v] < d + 1:
                depths[v] = d + 1

    fan_in = dict(G.in_degree())
    fan_out = dict(G.out_degree())

    # --- Features ---
    features: List[List[float]] = []
    for i in range(num_nodes):
        feat = [
            float(i),                # index
            float(node_types[i]),    # type
            float(topo_order.index(i)),
            float(depths[i]),
            float(fan_in.get(i, 0)),
            float(fan_out.get(i, 0)),
        ] + [0.0] * padding_length
        features.append(feat)

    # --- Build DGL graph ---
    if edges:
        src, dst = zip(*edges)
        g = dgl.graph((list(src), list(dst)), num_nodes=num_nodes)
    else:
        g = dgl.graph(([], []), num_nodes=num_nodes)

    g.ndata["feat"] = torch.tensor(features, dtype=torch.float)

    # --- Consistency: inputs must be the first num_pis nodes and type==0 ---
    is_pi_front = int((g.ndata["feat"][:num_pis, 1] == 0).all().item())
    if not is_pi_front:
        print(f"[SKIPPED] '{circuit_name}' failed PI-front assertion.")
        return None, None, circuit_name

    # --- Labels ---
    labels_tensor = None
    if label_dir is not None:
        ordering = read_variable_ordering(label_dir, circuit_name)
        if not ordering:
            print(f"[SKIPPED] '{circuit_name}' missing/empty ordering.")
            return None, None, circuit_name

        # Convert 1-based -> 0-based if looks 1-based
        if min(ordering) >= 1 and max(ordering) <= num_pis:
            ordering = [x - 1 for x in ordering]

        # Must be a permutation of 0..num_pis-1
        if len(ordering) != num_pis or sorted(ordering) != list(range(num_pis)):
            print(f"[SKIPPED] '{circuit_name}' invalid ordering (expect permutation 0..{num_pis-1}): {ordering}")
            return None, None, circuit_name

        labels_tensor = torch.tensor(ordering, dtype=torch.long)

    return g, labels_tensor, circuit_name


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert AIG edgelist (*.el) to DGLGraph dataset (.pt) with PI-front ordering")
    parser.add_argument("--el_folder", type=str, default="ISCAS85-graph", help="Folder containing *.el files")
    parser.add_argument("--label_dir", type=str, default="ISCAS85-graph", help="Folder for {circuit}_BDD_all/best_algorithm.txt")
    parser.add_argument("--save_dir", type=str, default="./processed", help="Output directory for .pt files")
    parser.add_argument("--padding_length", type=int, default=8, help="Zero padding length to mimic BLIF truth-table slot")
    parser.add_argument("--train", type=float, default=0.7, help="Train split")
    parser.add_argument("--val", type=float, default=0.2, help="Val split")
    parser.add_argument("--test", type=float, default=0.1, help="Test split")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    assert abs((args.train + args.val + args.test) - 1.0) < 1e-6, "Split ratios must sum to 1."

    el_files = glob.glob(os.path.join(args.el_folder, "*.el"))
    el_files.sort()
    if not el_files:
        print(f"[ERROR] No .el files found in {args.el_folder}")
        return

    graphs, labels, names = [], [], []
    for fp in el_files:
        print(f"Processing {os.path.basename(fp)} ...")
        g, lbl, name = build_graph_from_el(
            fp,
            padding_length=args.padding_length,
            label_dir=args.label_dir
        )
        if g is None or lbl is None:
            print(f"[INFO] Skipped {name}")
            continue
        # Final sanity: first num_inputs must be inputs
        num_inputs = int((g.ndata['feat'][:, 1] == 0).sum().item())
        if not int((g.ndata['feat'][:num_inputs, 1] == 0).all().item()):
            print(f"[INFO] Skipped {name} due to non-front inputs.")
            continue

        graphs.append(g)
        labels.append(lbl)
        names.append(os.path.basename(fp))

    if not graphs:
        print("[ERROR] No graphs processed successfully.")
        return

    # Shuffle & split
    random.seed(args.seed)
    combined = list(zip(graphs, labels, names))
    random.shuffle(combined)
    graphs[:], labels[:], names[:] = zip(*combined)

    n = len(graphs)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    data_train = {"graphs": list(graphs[:n_train]), "labels": list(labels[:n_train]), "circuit_names": list(names[:n_train])}
    data_val   = {"graphs": list(graphs[n_train:n_train+n_val]), "labels": list(labels[n_train:n_train+n_val]), "circuit_names": list(names[n_train:n_train+n_val])}
    data_test  = {"graphs": list(graphs[n_train+n_val:]), "labels": list(labels[n_train+n_val:]), "circuit_names": list(names[n_train+n_val:])}

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(data_train, os.path.join(args.save_dir, "aig_2_graph_Train_fix.pt"))
    torch.save(data_val,   os.path.join(args.save_dir, "aig_2_graph_Val_fix.pt"))
    torch.save(data_test,  os.path.join(args.save_dir, "aig_2_graph_Test_fix.pt"))

    print(f"\nSaved to {args.save_dir}:")
    print("  - aig_2_graph_Train_fix.pt")
    print("  - aig_2_graph_Va_fixl.pt")
    print("  - aig_2_graph_Test_fix.pt")


if __name__ == "__main__":
    main()