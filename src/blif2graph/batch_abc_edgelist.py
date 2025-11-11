#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import subprocess
from typing import Tuple, Optional

def run_abc_on_blif(abc_path: str, blif_path: str, out_el_path: str) -> bool:
    """
    Use ABC to read a BLIF, run 'strash', and export an edgelist to out_el_path.
    Returns True on success (based on process return code and existence of the target file).
    """
    os.makedirs(os.path.dirname(out_el_path), exist_ok=True)
    # Use -c to pass a command string; quote paths to avoid issues with spaces
    cmd = f'read_blif "{blif_path}"; strash; write_edgelist "{out_el_path}"'
    try:
        res = subprocess.run(
            [abc_path, "-c", cmd],
            capture_output=True,
            text=True,
            check=False
        )
        if res.returncode != 0:
            print(f"[ERROR] ABC failed: {blif_path}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}")
            return False
        if not os.path.exists(out_el_path):
            print(f"[ERROR] Exported .el file not found: {out_el_path}")
            return False
        return True
    except FileNotFoundError:
        print(f"[ERROR] ABC executable not found: {abc_path}. Please specify via --abc-path.")
        return False
    except Exception as e:
        print(f"[ERROR] Exception while running ABC: {e}")
        return False

def parse_el_file(el_path: str) -> Tuple[int, int]:
    """
    Parse a .el file:
    - Edge count = number of non-comment (#) valid lines
    - Node count = max of the first two integers across lines (ignores trailing 'AIG/PI/PO' and polarity bits)
    Returns (0, 0) if the file is empty or has no valid lines.
    """
    max_idx: Optional[int] = None
    edges = 0
    with open(el_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                # Non-standard line, ignore
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                # If the first two columns are not integers, skip this line
                continue
            edges += 1
            if max_idx is None or u > max_idx:
                max_idx = u
            if v > (max_idx if max_idx is not None else -1):
                max_idx = v
    node_size = max_idx if max_idx is not None else 0
    return node_size, edges

def main():
    parser = argparse.ArgumentParser(
        description="Batch-call ABC (read_blif->strash->write_edgelist) and report per-graph and average nodes/edges"
    )
    parser.add_argument(
        "--abc-path",
        default="./abc",
        help="Path to ABC executable (default: ./abc)"
    )
    parser.add_argument(
        "--blif-dir",
        default="/data/mingkai/coding_env/BDDs-Project/Dataset/ISCAS85_Blif",
        help="Directory containing .blif files"
    )
    parser.add_argument(
        "--out-dir",
        default="ISCAS85-graph",
        help="Output directory for .el files (relative or absolute path)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration if .el already exists"
    )
    args = parser.parse_args()

    blif_files = glob.glob(os.path.join(args.blif_dir, "*.blif"))
    blif_files.sort()
    if not blif_files:
        print(f"[WARN] No .blif files found in directory: {args.blif_dir}")
        return

    total_nodes = 0
    total_edges = 0
    graphs = 0

    print(f"[INFO] Output directory: {os.path.abspath(args.out_dir)}")
    print(f"[INFO] Found {len(blif_files)} BLIF files. Start processing...\n")

    for blif_path in blif_files:
        name = os.path.splitext(os.path.basename(blif_path))[0]
        out_el_path = os.path.join(args.out_dir, f"{name}.el")

        # Re-generate .el if needed
        if args.force or (not os.path.exists(out_el_path)):
            ok = run_abc_on_blif(args.abc_path, blif_path, out_el_path)
            if not ok:
                print(f"[SKIPPED] {name} failed to process. Skipping.\n")
                continue
        else:
            # .el exists, parse directly
            pass

        # Parse .el
        try:
            nodes, edges = parse_el_file(out_el_path)
        except Exception as e:
            print(f"[ERROR] Failed to parse {out_el_path}: {e}")
            continue

        print(f"[STATS] {name}: nodes={nodes}, edges={edges}")
        total_nodes += nodes
        total_edges += edges
        graphs += 1

    if graphs == 0:
        print("\n[RESULT] No graphs were processed successfully.")
        return

    avg_nodes = total_nodes / graphs
    avg_edges = total_edges / graphs

    print("\n[TOTAL] graphs={:d} | nodes={:d}, edges={:d}".format(graphs, total_nodes, total_edges))
    print("[AVERAGE] per-graph: nodes={:.2f}, edges={:.2f}".format(avg_nodes, avg_edges))

if __name__ == "__main__":
    main()