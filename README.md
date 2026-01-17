# InSiteDTA

A complex-free deep learning model for protein-ligand binding affinity prediction with intrinsic binding site detection.

**Key Features:**
- No molecular docking required
- No explicit binding site annotation needed
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

### 3. Install PyTorch and PyTorch Geometric

**Our tested environment:**
- Python: 3.9.19
- PyTorch: 2.5.1
- PyTorch Geometric: 2.6.1
- CUDA: 11.8

Install PyTorch based on your CUDA version from the official site:
https://pytorch.org/get-started/locally/

For our CUDA 11.8 setup:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

## Quick Start
```bash
python inference.py \
    --protein_pdb <path_to_protein.pdb> \
    --smiles <ligand_smiles>
```

**Example:**
```bash
python inference.py \
    --protein_pdb ./src/sample/1a30_protein.pdb \
    --smiles "CCO"
```

## Output

The model outputs predicted binding affinity in pK scale (higher values indicate stronger binding).

## Citation

TBD