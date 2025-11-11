BDD2Seq
Graph-to-sequence learning for BDD variable ordering in reversible-circuit synthesis. A GNN encoder plus a Pointer‑Network decoder optionally with Diverse Beam Search predicts input permutations that yield compact BDDs.

**Repository Layout**
- `data/`: BLIF datasets and derived artifacts for train/eval.
- `src/`:
  - `src_gat_ptr/`: PyTorch+DGL models, training, and inference.
  - `blif2graph/`: Converters BLIF/AIG/QASM → DGL graphs; ABC helpers.
  - `cudd_data_process/`: CUDD runners for classical reordering baselines.


**What This Code Does**
- Converts BLIF/AIG/QASM circuits into DGL graphs with structural features.
- Trains a GAT encoder + pointer decoder to generate input orderings.
- Runs inference via greedy or diverse beam search.

**Dependencies (high level)**
- Python 3.8+ with PyTorch and DGL.
- Optional: ABC and CUDD for baseline data generation (see `blif2graph/`, `cudd_data_process/`).
