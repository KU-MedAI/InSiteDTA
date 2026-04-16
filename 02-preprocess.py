# TODO: Implement flatten data structure
# TODO: Change to first-channel representation
# TODO: args.seed -> 0 for random to -1 for random

import os, argparse, pickle, json
import numpy as np
import pandas as pd
import torch

from datetime import datetime
from tqdm import tqdm
from typing import Literal
from sklearn.model_selection import train_test_split

from src.scripts.preprocess.generate_mol_object import (
    generate_mol_object,
    generate_conformers,
)
from src.scripts.preprocess.protein_voxelization import ProteinVoxelizer
from src.scripts.utils import print_args

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_dir",
        type=str,
        default="src/data/samples",
        help="Directory containing raw protein PDB files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="preprocessed_input",
        help="Output directory for preprocessed protein and ligand files",
    )

    # ligand featurization args
    parser.add_argument(
        "--smiles_csv",
        type=str,
        default="src/data/index/ligand_smiles_pdbbind2020.csv",
        help="Csv file path containing smiles with pdb id",
    )

    # protein voxelization args
    parser.add_argument(
        "--data_structure",
        type=str,
        default="nested",
        choices=["nested", "flatten"],
        help="Structure of the raw pdb data directory (pdbbind format: nested)",
    )

    parser.add_argument(
        "--voxel_size", type=int, default=2, help="Size of each voxel in Angstroms"
    )
    parser.add_argument(
        "--n_voxels",
        type=int,
        default=32,
        help="Number of voxels along each dimension of the 3D grid",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU index to use while protein voxelization",
    )
    parser.add_argument(
        "--label_radius",
        type=float,
        default=2.0,
        help="Radius for pocket labeling in voxelization (0 for exact voxel matching)",
    )
    parser.add_argument(
        "--center_method",
        type=str,
        default="intelligent",
        choices=["intelligent", "protein"],
        help="Voxel grid center method: 'intelligent' (shift toward pocket) or 'protein' (protein geometric center)",
    )
    parser.add_argument(
        "--ptn_dirname",
        type=str,
        default="input_protein",
        help="Directory name under save_dir for voxelized protein files",
    )
    parser.add_argument(
        "--lig_dirname",
        type=str,
        default="input_ligand",
        help="Directory name under save_dir for featurized ligand files",
    )

    # data split args
    parser.add_argument(
        "--seed",
        type=int,
        default=312,
        help="Seed for reproduce data split. 0 for random",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default="src/data/index/affinity_index_pdbbind2020.json",
        help="Json file containing binding affinity index of target proteins",
    )
    parser.add_argument(
        "--test_key_file",
        type=str,
        default="src/data/index/test_key_file_sample.txt",
        help="Path to file containing test set keys. Use 'none' for random split",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation",
    )

    args = parser.parse_args()
    return args


def collect_pdb_ids(
    raw_dir: str, data_structure: Literal["nested", "flatten"]
) -> list[str]:
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    if data_structure == "nested":
        pdb_id_ls = sorted(
            [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
        )
    elif data_structure == "flatten":
        pdb_id_ls = sorted(
            [
                "_".join(f.split("_")[:-1])
                for f in os.listdir(raw_dir)
                if f.endswith("protein.pdb")
            ]
        )

    print(f"0. Collected # {len(pdb_id_ls)} target pdb ids from raw_dir ({raw_dir})")
    return pdb_id_ls


def featurize_ligand(smiles_csv: str, pdb_id_ls: list[str], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    smiles_df = pd.read_csv(smiles_csv)

    for pdb_id in tqdm(pdb_id_ls, desc="1. Preparing ligands"):
        match_row = smiles_df[smiles_df["PDB_ID"] == pdb_id]
        if match_row.empty:
            print(f"[Warning] Skipping PDB ID '{pdb_id}' - not found in SMILES CSV")
            continue
        smi = match_row["Canonical SMILES"].iloc[0]
        out_name = f"{save_dir}/{pdb_id}_ligand.pkl"
        if os.path.exists(out_name):
            continue

        m = generate_mol_object(smi)
        if m is None:
            raise RuntimeError(f"Mol object was not created with smiles '{smi}'")
        m = generate_conformers(m, target_numConfs=5)

        with open(out_name, "wb") as fp:
            pickle.dump(m, fp)


def voxelize_protein(
    data_structure: Literal["nested", "flatten"],
    raw_dir: str,
    pdb_id_ls: list[str],
    save_dir: str,
    voxel_size: int = 2,
    n_voxels: int = 32,
    device=0,
    label_radius: float = 2.0,
    center_method: str = "intelligent",
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    # Path specification
    if data_structure == "nested":
        ptn_path_template = "{raw_dir}/{pdb_id}/{pdb_id}_protein.pdb"
        poc_path_template = "{raw_dir}/{pdb_id}/{pdb_id}_pocket.pdb"

    elif data_structure == "flatten":
        ptn_path_template = "{raw_dir}/{pdb_id}_protein.pdb"
        poc_path_template = "{raw_dir}/{pdb_id}_pocket.pdb"
        raise NotImplementedError("flatten mode is not supported yet")

    pv = ProteinVoxelizer(voxel_size=voxel_size, n_voxels=n_voxels)

    for pdb_id in tqdm(pdb_id_ls, desc="2. Preparing proteins"):
        ptn_path = ptn_path_template.format(raw_dir=raw_dir, pdb_id=pdb_id)
        poc_path = poc_path_template.format(raw_dir=raw_dir, pdb_id=pdb_id)
        out_voxel_name = f"{save_dir}/{pdb_id}_voxel.pkl"
        out_center_name = f"{save_dir}/{pdb_id}_center.pkl"

        if os.path.exists(out_voxel_name) and os.path.exists(out_center_name):
            continue

        defined_center = None
        if center_method == "protein":
            defined_center = pv.calc_protein_center(ptn_path)

        voxel, label, center = pv.voxelize_gpu_v2(
            protein_path=ptn_path,
            pocket_path=poc_path,
            r_cutoff=4.0,
            device=device,
            batch_size=8192,
            label_radius=label_radius,
            defined_center=defined_center,
        )

        # TODO: channel 위치 미리 조절
        protein_data = np.concatenate((voxel, label), axis=3).astype(np.float16)
        with open(out_voxel_name, "wb") as fp:
            pickle.dump(protein_data, fp)

        with open(out_center_name, "wb") as fp:
            pickle.dump(center, fp)


def split_data_keys(
    pdb_id_ls: list, val_size: float, seed: int, test_key_file: str = ""
):

    if test_key_file:
        with open(test_key_file, "r") as fp:
            ts_keys = fp.read().splitlines()
        tr_vl_keys = [i for i in pdb_id_ls if i not in ts_keys]
        print(
            f"  - Found {len(ts_keys)} test keys from {len(pdb_id_ls)} total targets. These will be excluded from training data."
        )
    else:
        tr_vl_keys, ts_keys = train_test_split(
            pdb_id_ls, test_size=val_size, random_state=seed, shuffle=True
        )
        print(
            f"  - Test keys were not specified. Will randomly split with the same size as validation ({val_size})."
        )

    tr_keys, vl_keys = train_test_split(
        tr_vl_keys, test_size=val_size, random_state=seed, shuffle=True
    )

    return tr_keys, vl_keys, ts_keys


def generate_data_cfg(
    pdb_id_ls: list,
    prep_dir_lig: str,
    prep_dir_ptn: str,
    save_dir: str,
    seed: int,
    index_file: str,
    val_size: float,
    test_key_file: str = "",
    voxel_size: int = 2,
    n_voxels: int = 32,
):
    print("3. Splitting data and generating data configuration files")
    out_cfg_name = os.path.join(
        save_dir, "data_config_" + datetime.now().strftime("%y%m%d-%H%M%S") + ".json"
    )

    data_cfg = {}

    data_cfg["created_at"] = datetime.now().strftime("%Y-%m-%d")
    data_cfg["seed"] = seed
    data_cfg["voxel_size"] = voxel_size
    data_cfg["n_voxels"] = n_voxels
    data_cfg["index_file"] = index_file

    data_cfg["vox_dir"] = prep_dir_ptn
    data_cfg["lig_dir"] = prep_dir_lig

    data_cfg["val_size"] = val_size
    data_cfg["tr_keys"] = None
    data_cfg["vl_keys"] = None
    data_cfg["ts_keys"] = None

    tr_keys, vl_keys, ts_keys = split_data_keys(
        pdb_id_ls, val_size, seed, test_key_file
    )
    data_cfg["tr_keys"] = tr_keys
    data_cfg["vl_keys"] = vl_keys
    data_cfg["ts_keys"] = ts_keys

    with open(out_cfg_name, "w") as fp:
        json.dump(data_cfg, fp, ensure_ascii=False, indent=4)
        print(f"  - Data configuration file successfully generated: {out_cfg_name}")

    print(f"4. Data split completed - Train: {len(tr_keys)}, Validation: {len(vl_keys)}, Test: {len(ts_keys)}")
    return data_cfg


def main():
    args = get_arguments()
    print_args(args)

    pdb_id_ls = collect_pdb_ids(args.raw_dir, args.data_structure)
    save_dir_lig = os.path.join(args.save_dir, args.lig_dirname)
    save_dir_ptn = os.path.join(args.save_dir, args.ptn_dirname)


    # featurize_ligand(
    #     smiles_csv=args.smiles_csv, pdb_id_ls=pdb_id_ls, save_dir=save_dir_lig
    # )

    voxelize_protein(
        data_structure=args.data_structure,
        raw_dir=args.raw_dir,
        pdb_id_ls=pdb_id_ls,
        save_dir=save_dir_ptn,
        voxel_size=args.voxel_size,
        n_voxels=args.n_voxels,
        label_radius=args.label_radius,
        center_method=args.center_method,
    )

    # generate_data_cfg(
    #     pdb_id_ls=pdb_id_ls,
    #     prep_dir_lig=save_dir_lig,
    #     prep_dir_ptn=save_dir_ptn,
    #     save_dir=args.save_dir,
    #     seed=args.seed,
    #     index_file=args.index_file,
    #     val_size=args.val_size,
    #     test_key_file=args.test_key_file,
    #     voxel_size=args.voxel_size,
    #     n_voxels=args.n_voxels,
    # )


if __name__ == "__main__":
    main()
