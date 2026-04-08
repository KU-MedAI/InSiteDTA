"""Run ColabFold protein structure prediction with MSA caching.

MSA files are stored in a separate directory (e.g., large disk).
On first run, MSAs are generated from FASTA and cached.
On subsequent runs, cached MSAs are reused automatically.

Usage:
python 02_run_colabfold.py --fasta ./colabfold_input.fasta --msa_dir /data/hawon/InSiteDTA_baselines/colabfold_msa --output_dir ./outputs --device 0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ColabFold with automatic MSA caching"
    )
    parser.add_argument("--fasta", type=str, required=True,
                        help="Input FASTA file (ColabFold format)")
    parser.add_argument("--msa_dir", type=str, required=True,
                        help="Directory to cache MSA files (can be on a separate disk)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for predicted structures")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--num_recycle", type=int, default=3)
    parser.add_argument("--model_type", type=str, default="alphafold2_multimer_v3")
    parser.add_argument("--msa_mode", type=str, default="mmseqs2_uniref_env",
                        choices=["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence"])
    parser.add_argument("--amber_relax", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def check_colabfold_installed() -> None:
    if shutil.which("colabfold_batch") is None:
        sys.exit("[ERROR] colabfold_batch not found. Install or activate conda env.")


def count_fasta_entries(fasta_path: str) -> int:
    with open(fasta_path) as f:
        return sum(1 for line in f if line.startswith(">"))


def msa_already_cached(msa_dir: str, n_expected: int) -> bool:
    """Check if MSA directory has enough .a3m files."""
    if not os.path.isdir(msa_dir):
        return False
    n_cached = len(list(Path(msa_dir).glob("*.a3m")))
    if n_cached >= n_expected:
        print(f"[MSA] Found {n_cached} cached MSAs in {msa_dir} (expected {n_expected})")
        return True
    if n_cached > 0:
        print(f"[MSA] Partial cache: {n_cached}/{n_expected} MSAs in {msa_dir}")
    return False


def generate_msa(fasta: str, msa_dir: str, msa_mode: str, device: int) -> float:
    """Run colabfold_batch --msa-only. Returns elapsed seconds."""
    os.makedirs(msa_dir, exist_ok=True)
    cmd = ["colabfold_batch", fasta, msa_dir, "--msa-only", "--msa-mode", msa_mode]
    print(f"\n[MSA] Generating MSAs → {msa_dir}")
    print(f"  Command: {' '.join(cmd)}")
    return _run(cmd, device)


def predict_structures(msa_dir: str, output_dir: str, args: argparse.Namespace) -> float:
    """Run colabfold_batch from cached MSAs. Returns elapsed seconds."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "colabfold_batch", msa_dir, output_dir,
        "--model-type", args.model_type,
        "--num-models", str(args.num_models),
        "--num-recycle", str(args.num_recycle),
    ]
    if args.amber_relax:
        cmd.append("--amber")

    print(f"\n[PREDICT] Running structure prediction → {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    return _run(cmd, args.device)


def _run(cmd: list[str], device: int) -> float:
    """Execute subprocess with GPU selection. Returns elapsed seconds."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    start = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[WARN] Command exited with code {result.returncode}")
    return elapsed


def count_predictions(output_dir: str) -> int:
    """Count unique PDB IDs from rank-1 unrelaxed PDBs."""
    pdb_ids = set()
    for f in Path(output_dir).glob("*_unrelaxed_rank_001_*.pdb"):
        pdb_ids.add(f.name.split("_unrelaxed_")[0])
    return len(pdb_ids)


def save_run_log(args: argparse.Namespace, n_input: int, n_predicted: int,
                 msa_time: float, predict_time: float) -> None:
    log = {
        "fasta": os.path.abspath(args.fasta),
        "msa_dir": os.path.abspath(args.msa_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "model_type": args.model_type,
        "num_models": args.num_models,
        "num_recycle": args.num_recycle,
        "device": args.device,
        "n_input": n_input,
        "n_predicted": n_predicted,
        "msa_seconds": round(msa_time, 1),
        "predict_seconds": round(predict_time, 1),
        "total_seconds": round(msa_time + predict_time, 1),
    }
    log_path = os.path.join(args.output_dir, "run_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def print_summary(n_input: int, n_predicted: int,
                  msa_time: float, predict_time: float, msa_cached: bool) -> None:
    total = msa_time + predict_time
    print(f"\n{'='*50}")
    print(f"  Input:       {n_input} entries")
    print(f"  Predicted:   {n_predicted}")
    print(f"  MSA:         {'cached (skipped)' if msa_cached else f'{msa_time/60:.1f} min'}")
    print(f"  Prediction:  {predict_time/60:.1f} min")
    print(f"  Total:       {total/60:.1f} min ({total/3600:.1f} h)")
    if n_predicted < n_input:
        print(f"  [WARN] {n_input - n_predicted} entries missing")
    print(f"{'='*50}")


def main():
    args = parse_args()
    check_colabfold_installed()

    n_input = count_fasta_entries(args.fasta)
    print(f"Input: {args.fasta} ({n_input} entries)")

    msa_cached = msa_already_cached(args.msa_dir, n_input)
    msa_time = 0.0
    if not msa_cached:
        msa_time = generate_msa(args.fasta, args.msa_dir, args.msa_mode, args.device)

    predict_time = predict_structures(args.msa_dir, args.output_dir, args)

    n_predicted = count_predictions(args.output_dir)
    save_run_log(args, n_input, n_predicted, msa_time, predict_time)
    print_summary(n_input, n_predicted, msa_time, predict_time, msa_cached)


if __name__ == "__main__":
    main()