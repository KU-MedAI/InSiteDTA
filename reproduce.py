import argparse, os, pickle, json
import numpy as np
import pandas as pd

from typing import Literal
from tqdm import tqdm

import torch
from torch_geometric.data import Batch

from src.model.model import InSiteDTA
from src.preprocess.generate_mol_object import generate_mol_object, generate_conformers
from src.preprocess.ligand_featurization import encode_ligand_to_Data
from src.preprocess.protein_voxelization import ProteinVoxelizer
from src.scripts.utils import calc_metrics


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, choices=["crystal", "redocked", "p2rank"], required=True, help="Coreset types to evaluate InSiteDTA")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    return parser.parse_args()

def prep_ligand(smi_csv, input_dir="./model_input"):
    os.makedirs(f"{input_dir}/ligands", exist_ok=True)
    
    smi_df = pd.read_csv(smi_csv)
    
    # ligand preparation
    for _, rows in tqdm(smi_df.iterrows()):
        pdb_id = rows['PDB_ID']
        smi = rows['Canonical SMILES']
        out_path = f"{input_dir}/ligands/{pdb_id}_ligand.pkl"
        if os.path.exists(out_path):
            continue
        
        m = generate_mol_object(smi)
        if m is None: raise RuntimeError(f"Mol object was not created with smiles '{smi}'")
        m = generate_conformers(m, target_numConfs=5)
        
        with open(out_path, 'wb') as fp:
            pickle.dump(m, fp)

def prep_protein(data_dir, input_dir="./model_input", device="cuda:0"):
    os.makedirs(f"{input_dir}/proteins", exist_ok=True)
    pdb_id_ls = sorted(os.listdir(data_dir))
    for pdb_id in tqdm(pdb_id_ls):
        pv = ProteinVoxelizer(voxel_size=2, n_voxels=32)
        ptn_path = f"{data_dir}/{pdb_id}/{pdb_id}_protein.pdb"
        poc_path = f"{data_dir}/{pdb_id}/{pdb_id}_pocket.pdb"
        out_data_name = os.path.join(f"{input_dir}/proteins/{pdb_id}_voxel_label_dim22.pkl")
        out_center_name = os.path.join(f"{input_dir}/proteins/{pdb_id}_center_coords.pkl")
        
        if os.path.exists(out_data_name) and os.path.exists(out_center_name):
            continue
        
        voxel, label, center = pv.voxelize_gpu_v2(
                            protein_path=ptn_path,
                            pocket_path=poc_path,
                            r_cutoff=4.0,
                            device=device,
                            batch_size=8192
                        )
        
        protein_data = np.concatenate((voxel, label), axis=3).astype(np.float16)
        with open(out_data_name, "wb") as fp:
            pickle.dump(protein_data, fp)

        with open(out_center_name, "wb") as fp:
            pickle.dump(center, fp)

def inference(lig_dir="./model_input/ligands", ptn_dir="./model_input/proteins", device="cuda:0", batch_size=128, index=None, ckpt=None):
    _get_paths = lambda x: [os.path.join(x, f) for f in sorted(os.listdir(x)) if f.endswith("_ligand.pkl") or f.endswith("_dim22.pkl")]
    _crop_ids = lambda x: os.path.basename(x).split("_")[0]
    
    lig_paths = _get_paths(lig_dir)
    ptn_paths = _get_paths(ptn_dir)
    
    # check whether lig/ptn keys matched
    lig_keys = [_crop_ids(l) for l in lig_paths]
    ptn_keys = [_crop_ids(p) for p in ptn_paths if p.endswith("")]
    
    # check validity
    if len(lig_keys) != len(ptn_keys):
        raise RuntimeError("Ligand/protein data size are not identical.")
    if any([lig_keys[i] != ptn_keys[i] for i in range(len(lig_keys))]):
        raise RuntimeError("Ligand/protein data key are not identical.")
    
    # load index
    total_target_ba = []
    with open(index, "r") as fp:
        index = json.load(fp)
    for p in ptn_keys:
        total_target_ba.append(index[p])
        
    # ligand load & featurization
    lig_feat_ls = []
    for lig in lig_paths:
        with open(lig, "rb") as fp:
            m = pickle.load(fp)
            lig_feat_ls.append(encode_ligand_to_Data(m))

    lig_batch_ls = []
    for i in range(0, len(lig_feat_ls), batch_size):
        lig_batch = lig_feat_ls[i: i+batch_size]
        lig_batch = Batch.from_data_list(lig_batch).to(device)
        lig_batch_ls.append(lig_batch)
        
    # protein load
    ptn_feat_ls = []
    for ptn in ptn_paths:
        with open(ptn, "rb") as fp:
            ptn_feat = pickle.load(fp).astype(np.float32)
            ptn_feat_ls.append(ptn_feat)
            
    ptn_batch_ls = []
    for i in range(0, len(ptn_feat_ls), batch_size):
        ptn_batch = np.stack(ptn_feat_ls[i: i+batch_size])
        ptn_batch = ptn_batch.astype(np.float32)
        ptn_batch = torch.from_numpy(ptn_batch).to(device)
        ptn_batch = ptn_batch[..., :21].permute(0, 4, 1, 2, 3)
        ptn_batch_ls.append(ptn_batch)
    
    model = InSiteDTA(out_channels=1)
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    model.to(device)
    model.eval()
    target_ba = [] if index is not None else None
    pred_ba_ls = []
    
    with torch.no_grad():
        for lig_batch, ptn_batch in zip(lig_batch_ls, ptn_batch_ls):
            pred_pocket, pred_ba = model(ptn_batch, lig_batch)
            pred_ba_ls.append(pred_ba)
            
    total_pred_ba = torch.concat(pred_ba_ls).cpu()
    return total_pred_ba, total_target_ba

def main():
    args = get_arguments()
    data = args.data
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    batch_size = 48
    index = "./src/data/PDBbind_aff_index.json"
    ckpt_ls = [
        "./src/model/ckpt/run_1.pt",
        "./src/model/ckpt/run_2.pt",
        "./src/model/ckpt/run_3.pt"
    ]
    data_dir = f"./src/data/coreset_{data}"
    smi_csv = "./src/data/coreset_lig_smiles.csv"
    input_dir = f"./model_input_{data}"
    
    ### main functions
    prep_ligand(smi_csv=smi_csv, input_dir=input_dir)
    prep_protein(data_dir=data_dir, input_dir=input_dir, device=device)
    for i, ckpt in enumerate(ckpt_ls):
        print(f"Evaluating {i+1}th model for coreset_{data}...")
        pred, target = inference(lig_dir=f"{input_dir}/ligands", ptn_dir=f"{input_dir}/proteins", batch_size=batch_size, device=device, index=index, ckpt=ckpt)
        calc_metrics(pred, target)
    
if __name__ == "__main__":
    main()