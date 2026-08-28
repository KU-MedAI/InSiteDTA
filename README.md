# InSiteDTA
<img width="4200" height="2657" alt="fig_overview" src="https://github.com/user-attachments/assets/2cf3e034-78f4-4b77-b86b-02bf7b33f9fe" />


A complex-free deep learning model for protein-ligand binding affinity prediction with internal binding site detection.

**Key Features:**

- Complex-free design: no molecular docking required
- Internal binding site detection: no explicit binding site input required
- Robust performance on imperfect structural inputs

## Installation

### 1. Clone repository

```bash
git clone https://github.com/KU-MedAI/InSiteDTA.git
cd InSiteDTA
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate insite
```

**Our tested environment:**

- Python: 3.9.19
- PyTorch: 2.5.1
- PyTorch Geometric: 2.6.1
- CUDA: 11.8

## Quick Start Example

```bash
python 01-inference.py \
    --pdb_path ./src/data/samples/4gkm/4gkm_protein.pdb \
    --smiles "Cc1ccc(c(c1)C(=O)[O-])Nc1ccccc1C(=O)[O-]" \
    --ckpt ./src/ckpt/CleanSplit_fold0_s312_teacher.pt
```

> Optional: `--save_bs_pdb <path>` exports the predicted binding-site residues as a PDB (`--save_voxel_pdb <path>` for the raw voxel grid; `--bs_threshold` tunes the probability cutoff).

## Training With Your Own Data

### Step 1: Prepare Data Structure

Organize your data in nested structure (PDBbind format):

```
raw_data/
├── {pdb_id}/
│   ├── {pdb_id}_protein.pdb
│   └── {pdb_id}_pocket.pdb
...
```

Prepare SMILES CSV file (`smiles.csv`):

```csv
PDB_ID,Canonical SMILES
1abc,CCO
1def,c1ccccc1
```

For affinity prediction, prepare affinity index JSON (`affinity.json`):

```json
{"1abc": 5.2, "1def": 7.8}
```

> **Note:** Affinity labels are loaded from `--index_file` (default: `src/data/index/affinity_index_pdbbind2020.json`). Samples without a matching entry are trained without affinity supervision.

### Step 2: Preprocess

```bash
python 02-preprocess.py \
    --raw_dir ./raw_data \
    --save_dir ./preprocessed \
    --smiles_csv ./smiles.csv \
    --index_file ./affinity.json \
    --test_key_file none \
    --voxel_size 2 \
    --n_voxels 32 \
    --device 0
```

`02-preprocess.py` generates ligand and protein inputs:

- `./preprocessed/input_ligand/{pdb_id}_ligand.pkl`
- `./preprocessed/input_protein/{pdb_id}_voxel.pkl`
- `./preprocessed/input_protein/{pdb_id}_center.pkl`
- `./preprocessed/data_config_YYMMDD-HHMMSS.json`

Set `--test_key_file` to a text file containing one test PDB ID per line, or use
`none` to create reproducible random validation and test splits using `--seed`.

### Step 3: Train

```bash
python 03-train.py \
    --data_config ./preprocessed/data_config_*.json \
    --save_dir ./checkpoints \
    --device 0 \
    --epochs 300 \
    --batch_size 48
```

Training uses **self-distillation** (an EMA teacher) and saves two checkpoints to `--save_dir`:

- `{split}_s{seed}_{timestamp}.pt` — student
- `{split}_s{seed}_{timestamp}_teacher.pt` — EMA teacher (**use this for inference / evaluate / reproduce**)

A `{split}_s{seed}_{timestamp}_results.json` with student/teacher metrics is also written.

## Evaluate Your Trained Model

```bash
python 04-evaluate.py \
    --ckpt ./checkpoints/{experiment_name}_teacher.pt \
    --result_file ./checkpoints/{experiment_name}_results.json \
    --save_dir ./evaluation \
    --use_tta \
    --device 0
```

`--use_tta` enables 6-face TTA. Omit it to run single-orientation evaluation.

The script will:

1. Load the test split defined in the training result file
2. Run 6-face test-time augmentation and average the affinity predictions and spatially aligned pocket logits
3. Report performance metrics (PCC, RMSE, MAE, DCC, DCC_SR, DVO)
4. Save detailed results to `{save_dir}/{experiment_name}_test_results.csv`

## Reproduce Paper Results

Run evaluation across the benchmark scenarios (`--scenario`: `crystal`, `redocked`, `p2rank`, `alphafold`):

```bash
# Evaluate on Coreset_crystal with 6-face TTA
python 05-reproduce.py --ckpt src/ckpt/CleanSplit_*.pt --scenario crystal --use_tta --device 0

# Evaluate on Coreset_redocked with 6-face TTA
python 05-reproduce.py --ckpt src/ckpt/CleanSplit_*.pt --scenario redocked --use_tta --device 0

# Evaluate on Coreset_p2rank with 6-face TTA
python 05-reproduce.py --ckpt src/ckpt/CleanSplit_*.pt --scenario p2rank --use_tta --device 0

# Evaluate on Coreset_alphafold with 6-face TTA
python 05-reproduce.py --ckpt src/ckpt/CleanSplit_*.pt --scenario alphafold --use_tta --device 0
```

The script will:

1. Prepare ligand features from SMILES
2. Voxelize protein structures
3. Evaluate each provided checkpoint with 6-face TTA (`--ckpt` accepts multiple, e.g. multiple seeds)
4. Report aggregated metrics — mean ± std (PCC, RMSE, MAE, DCC, DCC_SR, DVO)

## Output

**Inference (01-inference.py):**

- Predicted binding affinity in pK scale (higher values = stronger binding)

**Training (03-train.py):**

- Student checkpoint: `{save_dir}/{split}_s{seed}_{timestamp}.pt`
- EMA teacher checkpoint: `{save_dir}/{split}_s{seed}_{timestamp}_teacher.pt` (used for inference / evaluate / reproduce)
- Training results: `{save_dir}/{split}_s{seed}_{timestamp}_results.json`

**Evaluate (04-evaluate.py):**

- Evaluation results CSV: `{save_dir}/{experiment_name}_test_results.csv`

**Reproduce (05-reproduce.py):**

- Aggregated metrics across the provided checkpoints (mean ± std): PCC, RMSE, MAE, DCC, DCC_SR, DVO

## Data

**$Coreset_{crystal}$**

- Standard benchmark dataset from PDBbind

**$Coreset_{redocked}$**

- Coreset with redocked ligand in the native pocket

**$Coreset_{p2rank}$**

- Ligand redocked into the pocket predicted by P2Rank
([Krivák & Hoksza, 2018](https://doi.org/10.1186/s13321-018-0285-8))

**$Coreset_{alphafold}$**

- Protein structures predicted by ColabFold
([Mirdita et al., 2022](https://doi.org/10.1038/s41592-022-01488-1))
using AlphaFold-Multimer
([Evans et al., 2022](https://doi.org/10.1101/2021.10.04.463034))
(imperfect-structure benchmark)

## Citation

TBD
