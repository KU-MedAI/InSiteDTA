import argparse, os, pickle, json
import numpy as np
import pandas as pd

from typing import Literal
from tqdm import tqdm

import torch
from torch_geometric.data import Batch

from src.scripts.model.model import InSiteDTA
from src.scripts.preprocess.generate_mol_object import generate_mol_object, generate_conformers
from src.scripts.preprocess.ligand_featurization import encode_ligand_to_Data
from src.scripts.preprocess.protein_voxelization import ProteinVoxelizer
from src.scripts.utils_inference import calc_metrics
from src.scripts.utils import print_args


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["crystal", "redocked", "p2rank", "boltz2", "alphafold"], required=True, help="Coreset scenario to evaluate")
    parser.add_argument("--ckpt", type=str, nargs="+", required=True, help="Path(s) to model checkpoint(s)")
    parser.add_argument("--batch_size", type=int, default=64, help="Bacth size for inference")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use")
    return parser.parse_args()


def prep_ligand(smi_csv, input_dir="./model_input"):
    os.makedirs(f"{input_dir}/ligands", exist_ok=True)
    smi_df = pd.read_csv(smi_csv)
    # ligand preparation
    for _, rows in tqdm(smi_df.iterrows(), total=len(smi_df), desc="1. Preparing ligands"):
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

def prep_protein(data_dir, input_dir="./model_input", device="cuda:0", scenario="crystal"):
    os.makedirs(f"{input_dir}/proteins_{scenario}", exist_ok=True)
    pdb_id_ls = sorted(os.listdir(data_dir))
    # protein preparation
    for pdb_id in tqdm(pdb_id_ls, desc="2. Preparing proteins"):
        pv = ProteinVoxelizer(voxel_size=2, n_voxels=32)
        ptn_path = f"{data_dir}/{pdb_id}/{pdb_id}_protein.pdb"
        poc_path = f"{data_dir}/{pdb_id}/{pdb_id}_pocket.pdb"
        out_data_name = os.path.join(f"{input_dir}/proteins_{scenario}/{pdb_id}_voxel.pkl")
        out_center_name = os.path.join(f"{input_dir}/proteins_{scenario}/{pdb_id}_center.pkl")
        
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

def inference(lig_dir="./model_input/ligands", ptn_dir="./model_input/proteins", device="cuda:0", batch_size=64, index=None, ckpt=None, desc=None):
    _get_paths = lambda x: [os.path.join(x, f) for f in sorted(os.listdir(x)) if f.endswith("_ligand.pkl") or f.endswith("_voxel.pkl")]
    _crop_ids = lambda x: os.path.basename(x).split("_")[0]
    
    lig_paths = _get_paths(lig_dir)
    ptn_paths = _get_paths(ptn_dir)
    
    lig_map = {_crop_ids(l): l for l in lig_paths}
    ptn_map = {_crop_ids(p): p for p in ptn_paths}

    common_keys = sorted(set(lig_map.keys()) & set(ptn_map.keys()))
    n_lig_only = len(lig_map) - len(common_keys)
    n_ptn_only = len(ptn_map) - len(common_keys)

    if not common_keys:
        raise RuntimeError("No matching ligand/protein pairs found.")
    if n_lig_only > 0 or n_ptn_only > 0:
        print(f"  [INFO] lig={len(lig_map)}, ptn={len(ptn_map)} → {len(common_keys)} paired ({n_lig_only} lig-only, {n_ptn_only} ptn-only skipped)")

    lig_paths = [lig_map[k] for k in common_keys]
    ptn_paths = [ptn_map[k] for k in common_keys]
    
    # load index
    total_target_ba = []
    with open(index, "r") as fp:
        index = json.load(fp)
    for p in common_keys:
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
        for lig_batch, ptn_batch in tqdm(zip(lig_batch_ls, ptn_batch_ls), total=len(lig_batch_ls), desc=desc):
            pred_pocket, pred_ba = model(ptn_batch, lig_batch)
            if pred_ba.dim() == 0:
                pred_ba = pred_ba.unsqueeze(0)
            pred_ba_ls.append(pred_ba)
            
    total_pred_ba = torch.concat(pred_ba_ls).cpu()
    
    return total_pred_ba, total_target_ba

def main():
    args = get_arguments()
    scenario = args.scenario
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"; args.device = device
    batch_size = args.batch_size
    print_args(args)
    
    index = "./src/data/index/affinity_index_pdbbind2020.json"
    ckpt_ls = args.ckpt

    smi_csv = "./src/data/index/ligand_smiles_coreset.csv"
    data_dir = f"./src/data/coreset_{scenario}"
    input_dir = f"./model_input"

    prep_ligand(smi_csv=smi_csv, input_dir=input_dir)
    prep_protein(data_dir=data_dir, input_dir=input_dir, device=device, scenario=scenario)

    aggr_results = {'pcc': [], 'rmse': [], 'mae': []}
    for i, ckpt in enumerate(ckpt_ls):
        desc = f"3-{i+1}. Evaluating InSiteDTA ({i+1}/{len(ckpt_ls)}) on coreset_{scenario}"
        pred, target = inference(lig_dir=f"{input_dir}/ligands", ptn_dir=f"{input_dir}/proteins_{scenario}", batch_size=batch_size, device=device, index=index, ckpt=ckpt, desc=desc)
        pcc, rmse, mae = calc_metrics(pred, target)
        aggr_results['pcc'].append(pcc)
        aggr_results['rmse'].append(rmse)
        aggr_results['mae'].append(mae)
    
    print("4. Aggregated results on 3 different random seeds:")
    for metric, score_ls in aggr_results.items():
        mean = np.array(score_ls).mean()
        std = np.array(score_ls).std(ddof=1)
        print(f"- {metric.upper()}: {round(mean, 3)} ± {round(std, 3)}")
    
if __name__ == "__main__":
    main()