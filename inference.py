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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to protein PDB file")
    parser.add_argument("--smiles", type=str, required=True, help="SMILES string of the ligand")
    parser.add_argument("--ckpt", type=str, default="./src/model/ckpt/run_2.pt", help="Path to model checkpoint file")
    return parser.parse_args()


def prep_single_ligand(smiles):
    m = generate_mol_object(smiles)
    if m is None: raise RuntimeError(f"Mol object was not created with smiles '{smiles}'")
    m = generate_conformers(m, target_numConfs=5)
    
    return m
            
# def prep_ligand(smi_csv, out_dir="./model_input"):
#     os.makedirs(f"{out_dir}/ligands", exist_ok=True)
    
#     smi_df = pd.read_csv(smi_csv)
    
#     # ligand preparation
#     for _, rows in tqdm(smi_df.iterrows()):
#         pdb_id = rows['PDB_ID']
#         smi = rows['Canonical SMILES']
#         out_path = f"{out_dir}/ligands/{pdb_id}_ligand.pkl"
#         if os.path.exists(out_path):
#             continue
        
#         m = generate_mol_object(smi)
#         if m is None: raise RuntimeError(f"Mol object was not created with smiles '{smi}'")
#         m = generate_conformers(m, target_numConfs=5)
        
#         with open(out_path, 'wb') as fp:
#             pickle.dump(m, fp)

def prep_single_protein(pdb_path: str, device="cuda:0"):
    pv = ProteinVoxelizer(voxel_size=2, n_voxels=32)
    voxel, center = pv.voxelize_inference(
        protein_path = pdb_path
    )
    return voxel, center

# def inference(lig_dir="./model_input/ligands", ptn_dir="./model_input/proteins", device="cuda:0", batch_size=128, index=None, ckpt=None):
#     _get_paths = lambda x: [os.path.join(x, f) for f in sorted(os.listdir(x)) if f.endswith("_ligand.pkl") or f.endswith("_dim22.pkl")]
#     _crop_ids = lambda x: os.path.basename(x).split("_")[0]
    
#     lig_paths = _get_paths(lig_dir)
#     ptn_paths = _get_paths(ptn_dir)
    
#     # check whether lig/ptn keys matched
#     lig_keys = [_crop_ids(l) for l in lig_paths]
#     ptn_keys = [_crop_ids(p) for p in ptn_paths if p.endswith("")]
    
#     # check validity
#     if len(lig_keys) != len(ptn_keys):
#         raise RuntimeError("Ligand/protein data size are not identical.")
#     if any([lig_keys[i] != ptn_keys[i] for i in range(len(lig_keys))]):
#         raise RuntimeError("Ligand/protein data key are not identical.")
    
#     # load index
#     total_target_ba = []
#     with open(index, "r") as fp:
#         index = json.load(fp)
#     for p in ptn_keys:
#         total_target_ba.append(index[p])
        
#     # ligand load & featurization
#     lig_feat_ls = []
#     for lig in lig_paths:
#         with open(lig, "rb") as fp:
#             m = pickle.load(fp)
#             lig_feat_ls.append(encode_ligand_to_Data(m))

#     lig_batch_ls = []
#     for i in range(0, len(lig_feat_ls), batch_size):
#         lig_batch = lig_feat_ls[i: i+batch_size]
#         lig_batch = Batch.from_data_list(lig_batch).to(device)
#         lig_batch_ls.append(lig_batch)
        
#     # protein load
#     ptn_feat_ls = []
#     for ptn in ptn_paths:
#         with open(ptn, "rb") as fp:
#             ptn_feat = pickle.load(fp).astype(np.float32)
#             ptn_feat_ls.append(ptn_feat)
            
#     ptn_batch_ls = []
#     for i in range(0, len(ptn_feat_ls), batch_size):
#         ptn_batch = np.stack(ptn_feat_ls[i: i+batch_size])
#         ptn_batch = ptn_batch.astype(np.float32)
#         ptn_batch = torch.from_numpy(ptn_batch).to(device)
#         ptn_batch = ptn_batch[..., :21].permute(0, 4, 1, 2, 3)
#         ptn_batch_ls.append(ptn_batch)
    
#     model = InSiteDTA(out_channels=1)
#     model.load_state_dict(torch.load(ckpt, weights_only=False))
#     model.to(device)
#     model.eval()
#     target_ba = [] if index is not None else None
#     pred_ba_ls = []
    
#     with torch.no_grad():
#         for lig_batch, ptn_batch in zip(lig_batch_ls, ptn_batch_ls):
#             pred_pocket, pred_ba = model(ptn_batch, lig_batch)
#             pred_ba_ls.append(pred_ba)
            
#     total_pred_ba = torch.concat(pred_ba_ls).cpu()
#     return total_pred_ba, total_target_ba

def inference_single(voxel, mol, ckpt, device):
    voxel = voxel.astype(np.float32)
    voxel = torch.from_numpy(voxel).unsqueeze(0).permute(0, 4, 1, 2, 3)
    voxel = voxel.to(device)
    
    lig_data = encode_ligand_to_Data(mol)
    lig_data = Batch.from_data_list([lig_data])
    lig_data = lig_data.to(device)
    
        
    model = InSiteDTA(out_channels=1)
    model.load_state_dict(torch.load(ckpt, weights_only=False))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        pred_poc, pred_aff = model(voxel, lig_data)
    
    return pred_aff, pred_poc


def main():
    args = get_arguments()
    voxel, center = prep_single_protein(args.pdb_path)
    mol = prep_single_ligand(smiles=args.smiles)
    pred_aff, pred_poc = inference_single(voxel, mol, ckpt=args.ckpt, device="cuda:0")
    print(f"Predicted Binding Affinity: {round(pred_aff.item(),4)} (pK)")
    
if __name__ == "__main__":
    main()