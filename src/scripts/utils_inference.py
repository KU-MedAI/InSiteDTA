import os
import subprocess
import time
import numpy as np
import pandas as pd
from typing import Union

try:
    import torch
except ImportError:
    class torch:
        Tensor = None

class P2RankRunner:
    def __init__(self, bin_path="./src/p2rank/prank"):
        self.bin_path = bin_path

    def run_p2rank(self, pdb_path: str, output_dir: str) -> str:
        """Run P2Rank on a single PDB file. Returns path to the predictions CSV."""
        os.makedirs(output_dir, exist_ok=True)

        ds_path = os.path.join(output_dir, "input.ds")
        with open(ds_path, "w") as f:
            f.write(os.path.abspath(pdb_path))

        subprocess.run(
            [self.bin_path, "predict", ds_path, "-o", output_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return os.path.join(output_dir, f"{os.path.basename(pdb_path)}_predictions.csv")

    def get_pocket_center(self, pred_csv: str) -> np.ndarray:
        """Extract top-ranked pocket center coordinates from P2Rank predictions CSV."""
        pred = pd.read_csv(pred_csv)
        pred.columns = [c.strip() for c in pred.columns]
        top = pred.sort_values("rank").iloc[0]
        return np.array([float(top["center_x"]), float(top["center_y"]), float(top["center_z"])])

    def p2rank_res_to_pdb(self, pred_csv: str, src_pdb: str, out_dir: str) -> str:
        """Convert P2Rank predictions CSV to a pocket PDB file (top-ranked pocket)."""
        os.makedirs(out_dir, exist_ok=True)

        pred = pd.read_csv(pred_csv)
        pred.columns = [c.strip() for c in pred.columns]
        residue_ids = pred.residue_ids[0].split()

        poc_lines = []
        with open(src_pdb, "r") as f:
            for line in f.read().splitlines():
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                for res_id in residue_ids:
                    chain, r_id = res_id.split("_")
                    if line[21] == chain and line[22:27].strip() == r_id.strip():
                        poc_lines.append(line)

        out_path = os.path.join(out_dir, os.path.basename(src_pdb).split("_")[0] + "_pocket.pdb")
        with open(out_path, "w") as f:
            f.write("\n".join(poc_lines))

        return out_path


def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        print("============================== Measuring time starts. ============================== ")
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print(f"============================== WorkingTime[{original_fn.__name__}]: {end_time-start_time} sec============================== ")
        return result
    return wrapper_fn

def calc_PCC(pred:np.ndarray, target:np.ndarray):
    return np.corrcoef(pred, target)[0, 1]

def calc_RMSE(pred:np.ndarray, target:np.ndarray):
    return np.sqrt(np.mean((pred-target)**2))

def calc_MAE(pred:np.ndarray, target:np.ndarray):
    return np.mean(np.abs(pred-target))

def calc_metrics(pred: Union[np.ndarray, torch.Tensor, list], target:Union[np.ndarray, torch.Tensor, list]):
    # Unifying types into ndarray
    pred = np.array(pred)
    target = np.array(target)
    
    if pred.shape != target.shape:
        raise ValueError("Shape not matching")
    
    pcc, rmse, mae = calc_PCC(pred, target), calc_RMSE(pred, target), calc_MAE(pred, target)

    print("[Results]")
    print("  PCC :", pcc.round(4))
    print("  RMSE:", rmse.round(4))
    print("  MAE :", mae.round(4))

    return pcc, rmse, mae