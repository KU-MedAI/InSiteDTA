import time
import numpy as np
from typing import Union

try:
    import torch
except ImportError:
    class torch:
        Tensor = None

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

    print("[RESULTS]")
    print("- PCC :", pcc.round(4))
    print("- RMSE:", rmse.round(4))
    print("- MAE :", mae.round(4), "\n")

    return pcc, rmse, mae