"""Post-training test evaluation using the original voxel-based protocol."""

import numpy as np
import torch
from tqdm import tqdm

from .utils_train import calc_DCC_with_logit, calc_DVO_with_logit


# Keep the views and pocket behavior used by the former 04-evaluate.py.
FACE_ROTATIONS = (
    lambda x: x,
    lambda x: x.flip(4),
    lambda x: x.transpose(2, 4),
    lambda x: x.transpose(2, 4).flip(4),
    lambda x: x.transpose(3, 4),
    lambda x: x.transpose(3, 4).flip(4),
)


@torch.no_grad()
def evaluate_test_set(model, ts_loader, train_cfg, voxel_size, device):
    """Average affinity over six views; score pockets in the original view.

    Pocket metrics retain the former evaluation script's definitions, including
    its handling of empty predictions. They differ from the rank-1/equivalent-
    pocket benchmark protocol in pocket_evaluation.py.
    """
    if len(ts_loader.dataset) == 0:
        return {"status": "skipped", "reason": "No usable test samples", "n_samples": 0}

    model = model.to(device)
    model.eval()
    predictions, targets = [], []
    dccs, dvos = [], []
    empty_count = 0
    n_samples = 0

    for sample in tqdm(ts_loader, desc="Test evaluation (teacher, TTA-6faces)"):
        voxel = sample["voxel"].to(device)
        pocket = sample["pocket_label"].to(device)
        ligand = sample["lig_data"].to(device)

        affinity_views = []
        for index, rotation in enumerate(FACE_ROTATIONS):
            pocket_view, affinity_view = model(rotation(voxel), ligand)
            affinity_views.append(affinity_view.reshape(-1))
            if index == 0:
                pred_pocket = pocket_view
        pred_affinity = torch.stack(affinity_views).mean(dim=0)

        dcc, empty_indices = calc_DCC_with_logit(
            pred_pocket, pocket, voxel_size, train_cfg["DCC_threshold"],
        )
        dvo = calc_DVO_with_logit(
            pred_pocket, pocket, train_cfg["DVO_threshold"],
        )
        dccs.extend(dcc.cpu().tolist())
        dvos.extend(dvo.cpu().tolist())
        empty_count += len(empty_indices)
        n_samples += voxel.size(0)
        predictions.append(pred_affinity.cpu().numpy())
        targets.append(sample["true_aff"].detach().cpu().numpy().reshape(-1))

    pred = np.concatenate(predictions)
    true = np.concatenate(targets)
    labeled = np.isfinite(true)
    pred, true = pred[labeled], true[labeled]
    n_labeled = int(labeled.sum())
    pcc = None
    if n_labeled >= 2 and np.std(pred) > 0 and np.std(true) > 0:
        pcc = float(np.corrcoef(pred, true)[0, 1])

    metrics = {
        "status": "completed",
        "n_samples": n_samples,
        "n_affinity_labels": n_labeled,
        "PCC": pcc,
        "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))) if n_labeled else None,
        "MAE": float(np.mean(np.abs(pred - true))) if n_labeled else None,
        "DCC": float(np.mean(dccs)) if dccs else 0.0,
        "DCC_SR": sum(d <= train_cfg["DCC_SR_threshold"] for d in dccs) / n_samples,
        "DVO": float(sum(dvos) / n_samples),
        "DCC_nan_count": empty_count,
    }
    print("\nTest results (saved teacher):")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    return metrics
