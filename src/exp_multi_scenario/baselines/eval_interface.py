"""
eval_interface.py

Shared utilities for all baseline eval_wrapper scripts.

Each eval_wrapper.py is expected to follow this pattern:

    def main():
        args         = parse_args(build_parser())
        split_cfg    = load_split_config(args.eval_config, args.split)
        data_dir     = resolve_data_dir(args.split, args.scenario)
        preds        = run_inference_all_ckpts(args, data_dir)   # model-specific
        result       = build_result(args, preds)
        save_result(result, args.output_json)

Output JSON schema:
{
    "model":      str,
    "split":      str,
    "scenario":   str,
    "ckpts_used": [str, ...],
    "metrics": {
        "PCC":  {"mean": float, "std": float},
        "RMSE": {"mean": float, "std": float},
        "MAE":  {"mean": float, "std": float}
    },
    "per_sample": [
        {"pdb_id": str, "pred": float, "true": float}
    ]
}
"""

import argparse
import json
import os
import numpy as np
from typing import Dict, List, Union


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser(model_name: str) -> argparse.ArgumentParser:
    """Return base ArgumentParser with standard args shared across all wrappers."""
    parser = argparse.ArgumentParser(description=f"Eval wrapper for {model_name}")
    parser.add_argument("--eval_config",  type=str, required=True,
                        help="Path to eval_config.yaml")
    parser.add_argument("--split",        type=str, required=True,
                        choices=["original", "cleansplit"],
                        help="Data split to evaluate on")
    parser.add_argument("--scenario",     type=str, required=True,
                        choices=["crystal", "redocked", "p2rank", "alphafold", "boltz2"],
                        help="Evaluation scenario")
    parser.add_argument("--output_json",  type=str, required=True,
                        help="Path to write result JSON")
    parser.add_argument("--device",       type=int, default=0,
                        help="GPU device ID")
    parser.add_argument("--batch_size",   type=int, default=64)
    return parser


def get_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_eval_config(config_path: str) -> dict:
    """Load eval_config.yaml."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split_config(eval_config: dict, split: str) -> dict:
    """Extract split-specific config (data_config path, affinity_index, smiles_csv)."""
    split_cfg = eval_config["split_configs"].get(split)
    if split_cfg is None:
        raise ValueError(f"Split '{split}' not found in eval_config.split_configs")
    return split_cfg


def load_model_config(eval_config: dict, model_name: str) -> dict:
    """Extract model-specific config (ckpts, batch_size)."""
    model_cfg = eval_config["model_configs"].get(model_name)
    if model_cfg is None:
        raise ValueError(f"Model '{model_name}' not found in eval_config.model_configs")
    return model_cfg


def resolve_ckpts(model_cfg: dict, split: str) -> List[str]:
    """Return list of checkpoint paths for the given model and split."""
    ckpts = model_cfg["ckpts"].get(split, [])
    if not ckpts:
        raise ValueError(f"No checkpoints configured for split='{split}'")
    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    return ckpts


def resolve_data_dir(split: str, scenario: str, repo_root: str = ".") -> str:
    """Resolve coreset data directory path from split and scenario."""
    data_dir = os.path.join(repo_root, "src", "data", f"coreset_{scenario}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return data_dir


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    preds: Union[np.ndarray, list],
    targets: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Compute PCC, RMSE, MAE between predictions and targets."""
    preds   = np.array(preds,   dtype=float).squeeze()
    targets = np.array(targets, dtype=float).squeeze()

    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}")

    pcc  = float(np.corrcoef(preds, targets)[0, 1])
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae  = float(np.mean(np.abs(preds - targets)))

    return {"PCC": pcc, "RMSE": rmse, "MAE": mae}


def aggregate_metrics(metrics_per_ckpt: List[dict]) -> Dict[str, dict]:
    """
    Aggregate metrics across multiple checkpoint runs.

    Args:
        metrics_per_ckpt: list of {"PCC": float, "RMSE": float, "MAE": float}

    Returns:
        {"PCC": {"mean": float, "std": float}, "RMSE": ..., "MAE": ...}
    """
    keys = metrics_per_ckpt[0].keys()
    return {
        k: {
            "mean": float(np.mean([m[k] for m in metrics_per_ckpt])),
            "std":  float(np.std( [m[k] for m in metrics_per_ckpt], ddof=1)),
        }
        for k in keys
    }


# ---------------------------------------------------------------------------
# Result building & saving
# ---------------------------------------------------------------------------

def build_result(
    model: str,
    split: str,
    scenario: str,
    ckpts_used: List[str],
    aggregated_metrics: Dict[str, dict],
    per_sample: List[dict],
) -> dict:
    """Construct the standard result dictionary."""
    return {
        "model":      model,
        "split":      split,
        "scenario":   scenario,
        "ckpts_used": ckpts_used,
        "metrics":    aggregated_metrics,
        "per_sample": per_sample,
    }


def save_result(result: dict, output_json: str) -> None:
    """Save result dict to JSON file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved → {output_json}")


def print_metrics(result: dict) -> None:
    """Pretty-print aggregated metrics."""
    m = result["model"]
    s = result["split"]
    sc = result["scenario"]
    print(f"\n[{m} | split={s} | scenario={sc}]")
    for metric, vals in result["metrics"].items():
        print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
