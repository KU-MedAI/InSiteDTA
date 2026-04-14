import argparse
import json
import os
import sys
import csv
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
_REPO_ROOT = os.path.abspath(os.path.join(_MODEL_DIR, "../../../"))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _MODEL_DIR)

from src.scripts.dataloader import MasterDataLoader
from model import InSiteDTA
from src.scripts.utils import print_args
from src.scripts.utils_train import (
    SoftDiceWithLogitsLoss,
    calc_DCC_with_logit,
    calc_DVO_with_logit,
)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the trained model file."
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to the results & log json file, located in the same directory as checkpoint files",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "../../results/02_improve_training"),
        help="Directory to save evaluation results.",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to use")
    return parser.parse_args()


def eval_model(
    model: nn.Module,
    ts_loader: torch.utils.data.DataLoader,
    train_cfg: dict,
    voxel_size: int,
    exp_name: str,
    save_dir: str,
    device: int,
    return_preds=False,
):
    print("start evaluation!\n")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    aff_criterion = nn.MSELoss()
    poc_criterion_bce = nn.BCEWithLogitsLoss()
    poc_criterion_softdice = SoftDiceWithLogitsLoss()

    poc_weight = train_cfg["poc_loss_weight"]
    aff_weight = train_cfg["aff_loss_weight"]

    ts_dataset_size = len(ts_loader.dataset)

    result_dict = {}

    # [TEST]
    model.eval()
    ts_poc_loss = []
    ts_aff_loss = []
    ts_total_loss = []
    ts_DCCs = []
    ts_DVOs = []
    ts_nan_indices = []
    ts_pred_aff_values = []
    ts_true_aff_values = []

    with torch.no_grad():
        for sample in tqdm(ts_loader):
            voxel, pocket, lig_data, true_aff = (
                sample["voxel"],
                sample["pocket_label"],
                sample["lig_data"],
                sample["true_aff"],
            )

            voxel, pocket, lig_data, true_aff = (
                voxel.to(device),
                pocket.to(device),
                lig_data.to(device),
                true_aff.to(device),
            )

            pred_poc, pred_aff = model(voxel, lig_data)

            batch_size = voxel.size(0)

            poc_bce_loss = poc_criterion_bce(pred_poc, pocket) * batch_size
            poc_softdice_loss = poc_criterion_softdice(pred_poc, pocket) * batch_size
            poc_loss = poc_bce_loss + poc_softdice_loss

            has_aff_labels = ~torch.isnan(true_aff).all()
            aff_loss = (
                aff_criterion(pred_aff, true_aff) * batch_size
                if has_aff_labels
                else torch.tensor(0.0, device=voxel.device)
            )

            total_loss = poc_weight * poc_loss + aff_weight * aff_loss

            # Calc metrics
            DCC, nan_index = calc_DCC_with_logit(
                pred_poc,
                pocket,
                voxel_size=voxel_size,
                threshold=train_cfg["DCC_threshold"],
            )

            DVO = calc_DVO_with_logit(
                pred_poc, pocket, threshold=train_cfg["DVO_threshold"]
            )

            ts_poc_loss.append(poc_loss.item())
            ts_aff_loss.append(aff_loss.item())
            ts_total_loss.append(total_loss.item())
            ts_DCCs += DCC.tolist()
            ts_DVOs += DVO.tolist()
            ts_nan_indices += nan_index

            # When last batch size is 1
            if pred_aff.dim() == 0 or (pred_aff.dim() == 1 and pred_aff.size(0) == 1):
                ts_pred_aff_values.append(pred_aff.detach().cpu().view(1))
                ts_true_aff_values.append(true_aff.cpu().view(1))
            else:
                ts_pred_aff_values.append(pred_aff.detach().cpu())
                ts_true_aff_values.append(true_aff.cpu())
                ############ end of the inference ############

    avg_ts_poc_loss = sum(ts_poc_loss) / ts_dataset_size
    avg_ts_aff_loss = sum(ts_aff_loss) / ts_dataset_size
    avg_ts_total_loss = sum(ts_total_loss) / ts_dataset_size
    avg_ts_DCC = sum(ts_DCCs) / len(ts_DCCs) if ts_DCCs else 0
    avg_ts_DCC_SR = (
        len([DCC for DCC in ts_DCCs if DCC <= train_cfg["DCC_SR_threshold"]])
        / ts_dataset_size
    )
    avg_ts_DVO = sum(ts_DVOs) / ts_dataset_size
    ts_DCC_nan_count = len(ts_nan_indices)

    epoch_pred_aff_values = torch.cat(ts_pred_aff_values, dim=0).cpu().numpy().squeeze()
    epoch_true_aff_values = torch.cat(ts_true_aff_values, dim=0).cpu().numpy().squeeze()

    result_dict["pred_ba"] = epoch_pred_aff_values
    result_dict["true_ba"] = epoch_true_aff_values
    result_dict["DVO"] = ts_DVOs
    result_dict["DCC"] = ts_DCCs

    # Calculate correlation and error metrics
    ts_PCC = np.corrcoef(epoch_pred_aff_values, epoch_true_aff_values)[0, 1]
    ts_RMSE = np.sqrt(np.mean((epoch_pred_aff_values - epoch_true_aff_values) ** 2))
    ts_MAE = np.mean(np.abs(epoch_pred_aff_values - epoch_true_aff_values))

    # Print test results
    print(f"\nEvaluation completed. Experiment name: {exp_name}")
    print(f"Results: ")
    print(
        f"  Test Loss (pocket/affinity): {avg_ts_total_loss:.4f} ({avg_ts_poc_loss:.4f}/{avg_ts_aff_loss:.4f})"
    )
    print(f"  PCC: {ts_PCC:.4f}")
    print(f"  RMSE: {ts_RMSE:.4f}")
    print(f"  MAE: {ts_MAE:.4f}")
    print(f"  DCC_{train_cfg['DCC_threshold']}: {avg_ts_DCC:.4f}")
    print(
        f"  DCC_{train_cfg['DCC_threshold']}_SR_{train_cfg['DCC_SR_threshold']}: {avg_ts_DCC_SR:.4f}"
    )
    print(f"  DVO_{train_cfg['DVO_threshold']}: {avg_ts_DVO:.4f}")
    print(f"  DCC_nan_count: {ts_DCC_nan_count}")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{exp_name}_test_results.csv")

    with open(csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Experiment Name", exp_name])
        writer.writerow(["Total Loss", f"{avg_ts_total_loss:.4f}"])
        writer.writerow(["Pocket Loss", f"{avg_ts_poc_loss:.4f}"])
        writer.writerow(["Binding Affinity Loss", f"{avg_ts_aff_loss:.4f}"])
        writer.writerow([f"DCC_{train_cfg['DCC_threshold']}", f"{avg_ts_DCC:.4f}"])
        writer.writerow(
            [
                f"DCC_{train_cfg['DCC_threshold']}_SR_{train_cfg['DCC_SR_threshold']}",
                f"{avg_ts_DCC_SR:.4f}",
            ]
        )
        writer.writerow([f"DVO_{train_cfg['DVO_threshold']}", f"{avg_ts_DVO:.4f}"])
        writer.writerow(["DCC_nan_count", f"{ts_DCC_nan_count}"])
        writer.writerow(["PCC", f"{ts_PCC:.4f}"])
        writer.writerow(["RMSE", f"{ts_RMSE:.4f}"])
        writer.writerow(["MAE", f"{ts_MAE:.4f}"])

    print(f"\nEvaluated results have been saved to {csv_path}")

    # Return metrics in a dictionary
    metrics = {
        "total_loss": avg_ts_total_loss,
        "poc_loss": avg_ts_poc_loss,
        "aff_loss": avg_ts_aff_loss,
        "PCC": ts_PCC,
        "RMSE": ts_RMSE,
        "MAE": ts_MAE,
        "DCC": avg_ts_DCC,
        "DCC_SR": avg_ts_DCC_SR,
        "DVO": avg_ts_DVO,
        "DCC_nan_count": ts_DCC_nan_count,
    }

    if return_preds:
        return metrics, result_dict

    return metrics


def main():
    args = get_arguments()
    print_args(args)

    exp_name = os.path.basename(args.ckpt).replace(".pt", "")

    with open(args.result_file, "r") as fp:
        result_file = json.load(fp)

    train_cfg = result_file["train_config"]
    data_cfg = result_file["data_config"]

    mdl = MasterDataLoader(
        data_cfg, train_cfg["seed"], train_cfg["batch_size"], train_cfg["num_workers"]
    )
    ts_loader = mdl.get_ts_loader()

    model = InSiteDTA(
        spatial_dims=3,
        in_channels=train_cfg["in_channels"],
        out_channels=train_cfg["out_channels"],
        feature_size=train_cfg["feature_size"],
        sw_depths=train_cfg["sw_depths"],
        sw_window_size=train_cfg["sw_window_size"],
        sw_patch_size=train_cfg["sw_patch_size"],
        sw_num_heads=train_cfg["sw_num_heads"],
        sw_mlp_ratio=train_cfg["sw_mlp_ratio"],
        sw_qkv_bias=train_cfg["sw_qkv_bias"],
        sw_drop_rate=train_cfg["sw_drop_rate"],
        sw_attn_drop_rate=train_cfg["sw_attn_drop_rate"],
        sw_drop_path_rate=train_cfg["sw_drop_path_rate"],
        sw_act=train_cfg["sw_act"],
        mol_encoder_types=train_cfg["mol_encoder_types"],
        mol_in_channels=train_cfg["mol_in_channels"],
        mol_hidden_channels=train_cfg["mol_hidden_channels"],
        mol_out_channels=train_cfg["mol_out_channels"],
        mol_num_layers=train_cfg["mol_num_layers"],
        mol_num_interactions_3d=train_cfg["mol_num_interactions_3d"],
        mol_dropout_3d=train_cfg["mol_dropout_3d"],
        mol_cutoff_3d=train_cfg["mol_cutoff_3d"],
        mol_num_filters_schnet=train_cfg["mol_num_filters_schnet"],
        mol_edge_num_gaussian_schnet=train_cfg["mol_edge_num_gaussian_schnet"],
        mol_edge_num_fourier_feats_egnn=train_cfg["mol_edge_num_fourier_feats_egnn"],
        mol_soft_edge_egnn=train_cfg["mol_soft_edge_egnn"],
        mol_act=train_cfg["mol_act"],
        dec_drop_rate=train_cfg["dec_drop_rate"],
        dec_act=train_cfg["dec_act"],
        ca_num_heads=train_cfg["ca_num_heads"],
        ca_dropout=train_cfg["ca_dropout"],
    )

    model.load_state_dict(torch.load(args.ckpt, weights_only=True))

    metrics = eval_model(
        model,
        ts_loader,
        train_cfg,
        data_cfg["voxel_size"],
        exp_name,
        args.save_dir,
        args.device,
        return_preds=False,
    )

if __name__ == "__main__":
    main()
