"""Dock ligands into predicted protein structures using AutoDock Vina.

Step 1: SMILES → energy-minimized SDF → PDBQT (ligand)
Step 2: Predicted protein PDB → PDBQT (receptor)
Step 3: P2Rank pocket center → Vina config
Step 4: Run Vina docking (multiprocessing)
Step 5: Docked PDBQT → SDF → coreset_alphafold/{pdb_id}/{pdb_id}_ligand.sdf

Usage:
    python 04_dock_ligands.py \
        --coreset_dir ./coreset_alphafold \
        --smiles_csv ./ligand_smiles_coreset.csv \
        --p2rank_preds_dir ./coreset_alphafold/_p2rank_work/preds \
        --vina_path /path/to/vina \
        --num_processes 8
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from hw_toolkit.docking.box_calculator import DockingBoxCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dock ligands into predicted proteins via P2Rank + Vina"
    )
    parser.add_argument("--coreset_dir", type=str, required=True,
                        help="Coreset directory (e.g., ./prediction)")
    parser.add_argument("--smiles_csv", type=str, required=True,
                        help="CSV with columns: PDB_ID, Canonical SMILES")
    parser.add_argument("--p2rank_preds_dir", type=str, required=True,
                        help="P2Rank predictions directory (from 03_prepare_protein.py)")
    parser.add_argument("--vina_path", type=str, required=True,
                        help="Path to AutoDock Vina executable")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Working directory for intermediates (default: {coreset_dir}/_dock_work)")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--box_size", type=float, default=22.5,
                        help="Docking box size in Angstroms (default: 22.5)")
    return parser.parse_args()


# ─── Step 1: Prepare ligands (SMILES → SDF → PDBQT) ─────────────────────────

def initialize_mol(m):
    m = Chem.AddHs(m)
    result = AllChem.EmbedMolecule(m)
    if result != 0:
        raise RuntimeError(f"Conformer initialization failed. smiles: {Chem.MolToSmiles(m)}")
    return m


def minimize_mol(m):
    result = AllChem.MMFFOptimizeMolecule(m, confId=0)
    if result != 0:
        result = AllChem.MMFFOptimizeMolecule(m, confId=0, maxIters=1500)
        if result != 0:
            raise RuntimeError(f"Minimization failed. smiles: {Chem.MolToSmiles(m)}")
    return m


def prepare_ligands(smiles_csv: str, pdb_ids: list[str],
                    sdf_dir: str, pdbqt_dir: str) -> tuple[int, list[str]]:
    """SMILES → energy-minimized SDF → PDBQT.

    Returns:
        (n_success, list of failed pdb_ids)
    """
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(pdbqt_dir, exist_ok=True)

    smi_df = pd.read_csv(smiles_csv)
    smiles_map = dict(zip(smi_df["PDB_ID"], smi_df["Canonical SMILES"]))

    n_success = 0
    failed = []

    for pdb_id in pdb_ids:
        sdf_path = os.path.join(sdf_dir, f"{pdb_id}_ligand.sdf")
        pdbqt_path = os.path.join(pdbqt_dir, f"{pdb_id}_ligand.pdbqt")

        if pdb_id not in smiles_map:
            failed.append(pdb_id)
            continue

        # SDF generation (skip if exists)
        if not os.path.exists(sdf_path):
            try:
                m = Chem.MolFromSmiles(smiles_map[pdb_id])
                m = initialize_mol(m)
                m = minimize_mol(m)
                Chem.MolToMolFile(m, sdf_path)
            except Exception as e:
                print(f"  [WARN] {pdb_id}: SDF generation failed - {e}")
                failed.append(pdb_id)
                continue

        # PDBQT conversion (skip if exists)
        if not os.path.exists(pdbqt_path):
            cmd = f"mk_prepare_ligand.py -i {sdf_path} -o {pdbqt_path}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [WARN] {pdb_id}: PDBQT conversion failed")
                failed.append(pdb_id)
                continue

        n_success += 1

    print(f"[Step 1] Ligands prepared: {n_success} success, {len(failed)} failed")
    return n_success, failed


# ─── Step 2: Prepare receptors (PDB → PDBQT) ────────────────────────────────

def prepare_receptors(pdb_ids: list[str], coreset_dir: str,
                      pdbqt_dir: str) -> tuple[int, list[str]]:
    """Convert predicted protein PDBs to PDBQT format.

    Returns:
        (n_success, list of failed pdb_ids)
    """
    os.makedirs(pdbqt_dir, exist_ok=True)

    n_success = 0
    failed = []

    for pdb_id in pdb_ids:
        pdb_path = os.path.join(coreset_dir, pdb_id, f"{pdb_id}_protein.pdb")
        pdbqt_path = os.path.join(pdbqt_dir, f"{pdb_id}_protein.pdbqt")

        if os.path.exists(pdbqt_path):
            n_success += 1
            continue

        if not os.path.exists(pdb_path):
            failed.append(pdb_id)
            continue
        
        cmd = (f"prepare_receptor4.py -r {pdb_path} "
               f"-U nphs_lps_waters_deleteAltB -o {pdbqt_path}")
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] {pdb_id}: receptor preparation failed")
            failed.append(pdb_id)
            continue

        n_success += 1

    print(f"[Step 2] Receptors prepared: {n_success} success, {len(failed)} failed")
    return n_success, failed


# ─── Step 3: Generate Vina configs from P2Rank pocket centers ────────────────

def generate_vina_configs(pdb_ids: list[str], p2rank_preds_dir: str,
                          config_dir: str, box_size: float) -> tuple[int, list[str]]:
    """Generate Vina config files from P2Rank predicted pocket centers.

    Returns:
        (n_success, list of failed pdb_ids)
    """
    os.makedirs(config_dir, exist_ok=True)

    # Map pdb_id → prediction CSV
    pred_csvs = {}
    for f in os.listdir(p2rank_preds_dir):
        if f.endswith("_predictions.csv"):
            pdb_id = f.split("_protein")[0]
            pred_csvs[pdb_id] = os.path.join(p2rank_preds_dir, f)

    n_success = 0
    failed = []
    
    for pdb_id in pdb_ids:
        config_path = os.path.join(config_dir, f"{pdb_id}_config.txt")

        if os.path.exists(config_path):
            n_success += 1
            continue

        if pdb_id not in pred_csvs:
            failed.append(pdb_id)
            continue

        pred = pd.read_csv(pred_csvs[pdb_id])
        pred.columns = [c.strip() for c in pred.columns]

        if pred.empty:
            failed.append(pdb_id)
            continue

        center = [
            pred.loc[0, "center_x"].item(),
            pred.loc[0, "center_y"].item(),
            pred.loc[0, "center_z"].item(),
        ]

        dbc = DockingBoxCalculator()
        dbc.center = center
        dbc.box_size = [box_size, box_size, box_size]
        dbc.generate_autodock_config(config_path)

        n_success += 1

    print(f"[Step 3] Vina configs: {n_success} success, {len(failed)} failed")
    return n_success, failed


# ─── Step 4: Run Vina docking ────────────────────────────────────────────────

def _run_single_docking(args: tuple) -> tuple[str, bool]:
    """Single Vina docking job. Returns (pdb_id, success)."""
    lig, rec, cfg, vina_path, exhaustiveness, out_path = args
    pdb_id = os.path.basename(out_path).replace("_ligand.pdbqt", "")

    cmd = (f"{vina_path} --receptor {rec} --ligand {lig} "
           f"--config {cfg} --exhaustiveness {exhaustiveness} --out {out_path}")
    result = subprocess.run(cmd.split(), capture_output=True, text=True)

    return pdb_id, result.returncode == 0


def run_docking(pdb_ids: list[str], lig_pdbqt_dir: str, rec_pdbqt_dir: str,
                config_dir: str, docked_dir: str,
                vina_path: str, exhaustiveness: int,
                num_processes: int) -> tuple[int, list[str]]:
    """Run Vina docking with multiprocessing.

    Returns:
        (n_success, list of failed pdb_ids)
    """
    os.makedirs(docked_dir, exist_ok=True)

    args_list = []
    skipped = []

    for pdb_id in pdb_ids:
        lig = os.path.join(lig_pdbqt_dir, f"{pdb_id}_ligand.pdbqt")
        rec = os.path.join(rec_pdbqt_dir, f"{pdb_id}_protein.pdbqt")
        cfg = os.path.join(config_dir, f"{pdb_id}_config.txt")
        out = os.path.join(docked_dir, f"{pdb_id}_ligand.pdbqt")

        if os.path.exists(out):
            continue

        if not (os.path.exists(lig) and os.path.exists(rec) and os.path.exists(cfg)):
            skipped.append(pdb_id)
            continue

        args_list.append((lig, rec, cfg, vina_path, exhaustiveness, out))

    print(f"[Step 4] Docking {len(args_list)} complexes ({len(skipped)} skipped, "
          f"{num_processes} processes)...")

    failed = list(skipped)
    if args_list:
        start = time.time()
        with Pool(processes=num_processes) as pool:
            results = pool.map(_run_single_docking, args_list)
        elapsed = time.time() - start

        for pdb_id, success in results:
            if not success:
                failed.append(pdb_id)

        n_docked = sum(1 for _, s in results if s)
        print(f"  Docked: {n_docked}, Failed: {len(results) - n_docked}, "
              f"Time: {elapsed/60:.1f} min")

    n_total = sum(1 for p in pdb_ids
                  if os.path.exists(os.path.join(docked_dir, f"{p}_ligand.pdbqt")))
    print(f"[Step 4] Total docked PDBQTs: {n_total}")
    return n_total, failed


# ─── Step 5: Convert docked PDBQT → SDF and collect ─────────────────────────

def convert_and_collect(pdb_ids: list[str], docked_dir: str,
                        sdf_dir: str, coreset_dir: str) -> tuple[int, list[str]]:
    """Convert docked PDBQT to SDF and copy to coreset directory.

    Returns:
        (n_success, list of failed pdb_ids)
    """
    os.makedirs(sdf_dir, exist_ok=True)

    n_success = 0
    failed = []

    for pdb_id in pdb_ids:
        docked_pdbqt = os.path.join(docked_dir, f"{pdb_id}_ligand.pdbqt")
        sdf_path = os.path.join(sdf_dir, f"{pdb_id}_ligand.sdf")
        final_path = os.path.join(coreset_dir, pdb_id, f"{pdb_id}_ligand.sdf")

        if os.path.exists(final_path):
            n_success += 1
            continue

        if not os.path.exists(docked_pdbqt):
            failed.append(pdb_id)
            continue

        # PDBQT → SDF
        cmd = f"mk_export.py {docked_pdbqt} -s {sdf_path}"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(sdf_path):
            print(f"  [WARN] {pdb_id}: PDBQT→SDF conversion failed")
            failed.append(pdb_id)
            continue

        # Copy to final location
        os.makedirs(os.path.join(coreset_dir, pdb_id), exist_ok=True)
        shutil.copy2(sdf_path, final_path)
        n_success += 1

    print(f"[Step 5] Ligand SDFs collected: {n_success} success, {len(failed)} failed")
    return n_success, failed


# ─── Main ─────────────────────────────────────────────────────────────────────

def get_pdb_ids(coreset_dir: str) -> list[str]:
    """Get pdb_ids that have protein.pdb + pocket.pdb ready."""
    pdb_ids = []
    for d in sorted(os.listdir(coreset_dir)):
        if d.startswith("_"):
            continue
        full = os.path.join(coreset_dir, d)
        if not os.path.isdir(full):
            continue
        has_protein = os.path.exists(os.path.join(full, f"{d}_protein.pdb"))
        has_pocket = os.path.exists(os.path.join(full, f"{d}_pocket.pdb"))
        if has_protein and has_pocket:
            pdb_ids.append(d)
    return pdb_ids


def print_summary(pdb_ids: list[str], coreset_dir: str) -> None:
    n_protein = n_pocket = n_ligand = 0
    for p in pdb_ids:
        d = os.path.join(coreset_dir, p)
        if os.path.exists(os.path.join(d, f"{p}_protein.pdb")):
            n_protein += 1
        if os.path.exists(os.path.join(d, f"{p}_pocket.pdb")):
            n_pocket += 1
        if os.path.exists(os.path.join(d, f"{p}_ligand.sdf")):
            n_ligand += 1

    complete = sum(1 for p in pdb_ids
                   if all(os.path.exists(os.path.join(coreset_dir, p, f"{p}_{s}"))
                          for s in ["protein.pdb", "pocket.pdb", "ligand.sdf"]))

    print(f"\n{'='*50}")
    print(f"  PDB IDs with protein+pocket: {len(pdb_ids)}")
    print(f"  protein.pdb:  {n_protein}")
    print(f"  pocket.pdb:   {n_pocket}")
    print(f"  ligand.sdf:   {n_ligand}")
    print(f"  Complete:     {complete}/{len(pdb_ids)}")
    print(f"{'='*50}")


def main():
    args = parse_args()
    work_dir = args.work_dir or os.path.join(args.coreset_dir, "_dock_work")

    # Subdirectories for intermediates
    sdf_dir = os.path.join(work_dir, "00_ligand_sdf")
    lig_pdbqt_dir = os.path.join(work_dir, "01_ligand_pdbqt")
    rec_pdbqt_dir = os.path.join(work_dir, "01_receptor_pdbqt")
    config_dir = os.path.join(work_dir, "02_vina_configs")
    docked_dir = os.path.join(work_dir, "03_docked_pdbqt")
    docked_sdf_dir = os.path.join(work_dir, "04_docked_sdf")

    pdb_ids = get_pdb_ids(args.coreset_dir)
    if not pdb_ids:
        sys.exit("[ERROR] No complexes with protein.pdb + pocket.pdb found.")
    print(f"Found {len(pdb_ids)} complexes ready for docking.\n")

    prepare_ligands(args.smiles_csv, pdb_ids, sdf_dir, lig_pdbqt_dir)
    prepare_receptors(pdb_ids, args.coreset_dir, rec_pdbqt_dir)
    generate_vina_configs(pdb_ids, args.p2rank_preds_dir, config_dir, args.box_size)
    run_docking(pdb_ids, lig_pdbqt_dir, rec_pdbqt_dir, config_dir, docked_dir,
                args.vina_path, args.exhaustiveness, args.num_processes)
    convert_and_collect(pdb_ids, docked_dir, docked_sdf_dir, args.coreset_dir)
    print_summary(pdb_ids, args.coreset_dir)


if __name__ == "__main__":
    main()
