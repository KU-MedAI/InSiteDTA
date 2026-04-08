"""Post-process ColabFold outputs: protein PDB collection + P2Rank pocket prediction.

Step 1: Collect ColabFold rank-1 PDBs → coreset_alphafold/{pdb_id}/{pdb_id}_protein.pdb
Step 2: Run P2Rank binding site prediction on collected proteins
Step 3: Convert P2Rank results → {pdb_id}_pocket.pdb

Usage:
    python 03_prepare_protein.py \
        --colabfold_dir ./outputs \
        --output_dir ./coreset_alphafold \
        --p2rank_path /path/to/prank
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process ColabFold outputs + P2Rank pocket prediction"
    )
    parser.add_argument("--colabfold_dir", type=str, required=True,
                        help="ColabFold output directory containing predicted PDBs")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (e.g., coreset_alphafold)")
    parser.add_argument("--p2rank_path", type=str, required=True,
                        help="Path to P2Rank executable (prank)")
    parser.add_argument("--p2rank_work_dir", type=str, default=None,
                        help="Working directory for P2Rank (default: {output_dir}/_p2rank_work)")
    return parser.parse_args()


# ─── Step 1: Collect ColabFold outputs ────────────────────────────────────────

def collect_colabfold_outputs(colabfold_dir: str, output_dir: str) -> list[str]:
    """Find rank-1 PDBs and copy to coreset_alphafold/{pdb_id}/{pdb_id}_protein.pdb.

    Returns:
        List of successfully collected pdb_ids.
    """
    pdb_files = sorted(Path(colabfold_dir).glob("*_unrelaxed_rank_001_*.pdb"))
    if not pdb_files:
        pdb_files = sorted(Path(colabfold_dir).glob("*_relaxed_rank_001_*.pdb"))

    collected = []
    for pdb_file in pdb_files:
        pdb_id = pdb_file.name.split("_unrelaxed_")[0].split("_relaxed_")[0]
        dest_dir = os.path.join(output_dir, pdb_id)
        dest_path = os.path.join(dest_dir, f"{pdb_id}_protein.pdb")

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(str(pdb_file), dest_path)
        # dest_path 텍스트를 열어서 맨 윗줄이 MODEL 로 시작하면 그 줄 삭제하고 파일 덮어쓰기
        with open(dest_path, 'r') as f:
            lines = f.readlines()
        if lines and lines[0].strip().startswith("MODEL"):
            with open(dest_path, 'w') as f:
                f.writelines(lines[1:])
        collected.append(pdb_id)

    print(f"[Step 1] Collected {len(collected)} protein PDBs → {output_dir}")
    return collected


# ─── Step 2: Run P2Rank ───────────────────────────────────────────────────────

def run_p2rank(pdb_ids: list[str], output_dir: str,
               p2rank_path: str, work_dir: str) -> str:
    """Run P2Rank batch prediction on collected protein PDBs.

    Returns:
        Path to P2Rank predictions directory.
    """
    os.makedirs(work_dir, exist_ok=True)
    preds_dir = os.path.join(work_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Write dataset file (list of PDB paths)
    pdb_paths = []
    for pdb_id in pdb_ids:
        pdb_path = os.path.abspath(os.path.join(output_dir, pdb_id, f"{pdb_id}_protein.pdb"))
        pdb_paths.append(pdb_path)

    ds_file = os.path.join(work_dir, "pdb_paths.ds")
    with open(ds_file, "w") as f:
        f.write("\n".join(pdb_paths))

    cmd = f"{p2rank_path} predict {ds_file} -o {preds_dir}"
    print(f"[Step 2] Running P2Rank on {len(pdb_ids)} proteins...")
    print(f"  Command: {cmd}")
    result = subprocess.run(cmd.split(), text=True)

    if result.returncode != 0:
        print(f"  [WARN] P2Rank exited with code {result.returncode}")

    return preds_dir


# ─── Step 3: Generate pocket PDBs ────────────────────────────────────────────

def p2rank_res_to_pocket_pdb(pred_csv: str, src_pdb: str, out_path: str) -> int:
    """Extract pocket residues from P2Rank prediction and write pocket.pdb.

    Args:
        pred_csv: P2Rank *_predictions.csv file
        src_pdb: Source protein PDB file
        out_path: Output pocket PDB path

    Returns:
        Number of pocket residue lines written.
    """
    pred = pd.read_csv(pred_csv)
    pred.columns = [c.strip() for c in pred.columns]

    if pred.empty or pd.isna(pred.residue_ids.iloc[0]):
        return 0

    residue_ids = pred.residue_ids.iloc[0].split()

    # Parse residue_ids: format is "{chain}_{resnum}" e.g., "A_123"
    target_residues = set()
    for res_id in residue_ids:
        parts = res_id.split("_")
        if len(parts) >= 2:
            chain = parts[0]
            resnum = parts[1]
            target_residues.add((chain, resnum))

    pocket_lines = []
    with open(src_pdb) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21]
            resnum = line[22:27].strip()
            if (chain, resnum) in target_residues:
                pocket_lines.append(line)

    with open(out_path, "w") as f:
        for line in pocket_lines:
            f.write(line if line.endswith("\n") else line + "\n")
        f.write("END\n")

    return len(pocket_lines)


def generate_all_pocket_pdbs(pdb_ids: list[str], output_dir: str,
                             preds_dir: str) -> tuple[int, int]:
    """Generate pocket.pdb for all complexes from P2Rank results.

    Returns:
        (n_success, n_failed)
    """
    # Find all prediction CSVs
    pred_csvs = {}
    for f in os.listdir(preds_dir):
        if f.endswith("_predictions.csv"):
            pdb_id = f.split("_protein")[0]
            pred_csvs[pdb_id] = os.path.join(preds_dir, f)

    n_success, n_failed = 0, 0
    for pdb_id in pdb_ids:
        src_pdb = os.path.join(output_dir, pdb_id, f"{pdb_id}_protein.pdb")
        out_path = os.path.join(output_dir, pdb_id, f"{pdb_id}_pocket.pdb")

        if pdb_id not in pred_csvs:
            print(f"  [WARN] {pdb_id}: no P2Rank prediction found")
            n_failed += 1
            continue

        n_lines = p2rank_res_to_pocket_pdb(pred_csvs[pdb_id], src_pdb, out_path)
        if n_lines > 0:
            n_success += 1
        else:
            print(f"  [WARN] {pdb_id}: empty pocket (0 residues)")
            n_failed += 1

    print(f"[Step 3] Pocket PDBs: {n_success} success, {n_failed} failed")
    return n_success, n_failed


# ─── Main ─────────────────────────────────────────────────────────────────────

def print_summary(pdb_ids: list[str], output_dir: str) -> None:
    n_protein = sum(1 for p in pdb_ids
                    if os.path.exists(os.path.join(output_dir, p, f"{p}_protein.pdb")))
    n_pocket = sum(1 for p in pdb_ids
                   if os.path.exists(os.path.join(output_dir, p, f"{p}_pocket.pdb")))

    print(f"\n{'='*50}")
    print(f"  Total PDB IDs:   {len(pdb_ids)}")
    print(f"  protein.pdb:     {n_protein}")
    print(f"  pocket.pdb:      {n_pocket}")
    print(f"  Output dir:      {output_dir}")
    print(f"{'='*50}")


def main():
    args = parse_args()
    work_dir = args.p2rank_work_dir or os.path.join(args.output_dir, "_p2rank_work")

    pdb_ids = collect_colabfold_outputs(args.colabfold_dir, args.output_dir)
    if not pdb_ids:
        sys.exit("[ERROR] No ColabFold outputs found.")

    preds_dir = run_p2rank(pdb_ids, args.output_dir, args.p2rank_path, work_dir)
    generate_all_pocket_pdbs(pdb_ids, args.output_dir, preds_dir)
    print_summary(pdb_ids, args.output_dir)


if __name__ == "__main__":
    main()
