import numpy as np
import argparse
import json
import gc
import os
import csv
import torch
import torch.nn as nn
from types import SimpleNamespace
from tqdm import tqdm
import sys

from model.model import CustomSwinUnetr
sys.path.append("../../../../SCRIPTS")
from data.dataloader import MasterDataLoader
from _unspecified import calc_batch_vDCC_count_nan_with_logit, calc_batch_DVO_with_logit, DiceWithLogitsLoss, SoftDiceWithLogitsLoss, calc_f1_score_logit


def dict_to_args(d):
    # dictionary를 namespace로 변환
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_args(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_args(i) for i in d]
    else:
        return d


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to evaluate the model performance.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--ckpt_config', type=str, required=True, help="Path to the json config file containing settings the ckpt trained on.")
    parser.add_argument('--output_dir', type=str, default='../RESULTS/evaluation', help="Directory to save evaluation results.")
    return parser.parse_args()


def eval_model(model, ts_dataloader, train_config, data_config, experiment_name, save_dir, gpu, return_preds=False):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    # Initialize lists to store outputs and targets
    ts_recon_losses = []
    ts_pocket_losses = []
    ts_aff_losses = []
    ts_total_losses = []
    ts_vDCCs = []
    ts_DVOs = []
    ts_f1_scores = []
    ts_nan_indices = []
    ts_aff_all_preds = []
    ts_aff_all_targets = []

    # Loss functions
    recon_criterion = nn.MSELoss()
    pocket_dice_criterion = DiceWithLogitsLoss()
    pocket_bce_criterion = nn.BCEWithLogitsLoss()
    aff_criterion = nn.MSELoss()
    
    # Load loss weight
    recon_weight = getattr(train_config, "recon_loss_weight", 1)
    pocket_weight = getattr(train_config, "pocket_loss_weight", 2)
    aff_weight = getattr(train_config, "aff_loss_weight", 0.2)

    ts_dataset_size = len(ts_dataloader.dataset)
    result_dict = {}
    # Process batches
    with torch.no_grad():  # Add no_grad for inference to save memory
        for sample in tqdm(ts_dataloader):
            voxel, pocket, ligand_data, b_aff = sample["voxel"], sample["pocket_label"], sample["ligand_data"], sample["b_aff"]
            voxel, pocket, ligand_data, b_aff = voxel.to(device), pocket.to(device), ligand_data.to(device), b_aff.to(device)    

            output_voxel, output_aff = model(voxel, ligand_data)
            recon_voxel = output_voxel[:, :11, ...]
            pred_pocket = output_voxel[:, -1:, ...]

            # Calculate losses
            batch_size = voxel.size(0)
            recon_loss = torch.tensor([0]).to(device)
            pocket_dice_loss = pocket_dice_criterion(pred_pocket, pocket) * batch_size
            pocket_bce_loss = pocket_bce_criterion(pred_pocket, pocket) * batch_size
            pocket_loss = (pocket_dice_loss + pocket_bce_loss)
            aff_loss = aff_criterion(output_aff, b_aff) * batch_size
            
            total_loss = (
                recon_weight * recon_loss + 
                pocket_weight * pocket_loss +
                aff_weight * aff_loss
            )
            
            # Calculate metrics
            vDCC, nan_index = calc_batch_vDCC_count_nan_with_logit(pred_pocket, pocket, voxel_size=data_config["voxel_size"], threshold=train_config.DCC_threshold)
            DVO = calc_batch_DVO_with_logit(pred_pocket, pocket, threshold=train_config.DVO_threshold)
            f1 = calc_f1_score_logit(pred_pocket, pocket)
            
            # Append losses and metrics
            ts_recon_losses.append(recon_loss.item())
            ts_pocket_losses.append(pocket_loss.item())
            ts_aff_losses.append(aff_loss.item())
            ts_total_losses.append(total_loss.item())
            ts_vDCCs += vDCC.tolist()
            ts_DVOs += DVO.tolist()
            ts_f1_scores.append(f1)
            ts_nan_indices += nan_index

            # Last batch의 size가 1일 경우 처리
            if output_aff.dim() == 0 or (output_aff.dim() == 1 and output_aff.size(0) == 1):
                # 스칼라나 단일 요소 텐서 처리
                ts_aff_all_preds.append(output_aff.detach().cpu().view(1))
                ts_aff_all_targets.append(b_aff.cpu().view(1))
            else:
                # 일반적인 배치 처리
                ts_aff_all_preds.append(output_aff.detach().cpu())
                ts_aff_all_targets.append(b_aff.cpu())
                
            # Free memory
            del voxel, pocket, ligand_data, b_aff
            del output_voxel, output_aff, recon_voxel, pred_pocket
            del recon_loss, pocket_dice_loss, pocket_bce_loss, pocket_loss, aff_loss, total_loss
    
    # Calculate average metrics
    avg_ts_recon_loss = sum(ts_recon_losses) / ts_dataset_size
    avg_ts_pocket_loss = sum(ts_pocket_losses) / ts_dataset_size
    avg_ts_aff_loss = sum(ts_aff_losses) / ts_dataset_size
    avg_ts_total_loss = sum(ts_total_losses) / ts_dataset_size
    avg_ts_vDCC = sum(ts_vDCCs) / len(ts_vDCCs) if ts_vDCCs else 0
    avg_ts_vDCC_SR = len([DCC for DCC in ts_vDCCs if DCC <= train_config.DCC_SR_threshold]) / ts_dataset_size
    avg_ts_DVO = sum(ts_DVOs) / ts_dataset_size
    avg_ts_f1 = sum(ts_f1_scores) / len(ts_f1_scores) if ts_f1_scores else 0
    ts_DCC_nan_count = len(ts_nan_indices)            

    # Concatenate all predictions and targets
    epoch_preds = torch.cat(ts_aff_all_preds, dim=0).cpu().numpy().squeeze()
    epoch_targets = torch.cat(ts_aff_all_targets, dim=0).cpu().numpy().squeeze()            
    
    result_dict["pred_ba"] = epoch_preds
    result_dict["true_ba"] = epoch_targets
    result_dict["DVO"] = ts_DVOs
    result_dict["DCC"] = ts_vDCCs
    result_dict["F1"] = ts_f1_scores

    # Calculate correlation and error metrics
    ts_PCC = np.corrcoef(epoch_preds, epoch_targets)[0, 1]
    ts_RMSE = np.sqrt(np.mean((epoch_preds - epoch_targets) ** 2))
    ts_MAE = np.mean(np.abs(epoch_preds - epoch_targets))

    # Print test results
    print(f"Test Results for {experiment_name}:")
    print(f"Loss (total/recon/pocket/b_aff): {avg_ts_total_loss:.4f} ({avg_ts_recon_loss:.4f}/{avg_ts_pocket_loss:.4f}/{avg_ts_aff_loss:.4f})")
    print(f"vDCC_{train_config.DCC_threshold}: {avg_ts_vDCC:.4f}")
    print(f"vDCC_{train_config.DCC_threshold}_SR_{train_config.DCC_SR_threshold}: {avg_ts_vDCC_SR:.4f}")
    print(f"DVO_{train_config.DVO_threshold}: {avg_ts_DVO:.4f}")
    print(f"F1 Score: {avg_ts_f1:.4f}")
    print(f"DCC_nan_count: {ts_DCC_nan_count}")
    print(f"PCC: {ts_PCC:.4f}")
    print(f"RMSE: {ts_RMSE:.4f}")
    print(f"MAE: {ts_MAE:.4f}")

    # Save results to file
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'{experiment_name}_test_results.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Experiment Name', experiment_name])
        writer.writerow(['Total Loss', f'{avg_ts_total_loss:.4f}'])
        writer.writerow(['Reconstruction Loss', f'{avg_ts_recon_loss:.4f}'])
        writer.writerow(['Pocket Loss', f'{avg_ts_pocket_loss:.4f}'])
        writer.writerow(['Binding Affinity Loss', f'{avg_ts_aff_loss:.4f}'])
        writer.writerow([f'vDCC_{train_config.DCC_threshold}', f'{avg_ts_vDCC:.4f}'])
        writer.writerow([f'vDCC_{train_config.DCC_threshold}_SR_{train_config.DCC_SR_threshold}', f'{avg_ts_vDCC_SR:.4f}'])
        writer.writerow([f'DVO_{train_config.DVO_threshold}', f'{avg_ts_DVO:.4f}'])
        writer.writerow(['F1 Score', f'{avg_ts_f1:.4f}'])
        writer.writerow(['DCC_nan_count', f'{ts_DCC_nan_count}'])
        writer.writerow(['PCC', f'{ts_PCC:.4f}'])
        writer.writerow(['RMSE', f'{ts_RMSE:.4f}'])
        writer.writerow(['MAE', f'{ts_MAE:.4f}'])
    
    print(f"Results saved to {csv_path}")
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return metrics in a dictionary
    metrics = {
        'total_loss': avg_ts_total_loss,
        'recon_loss': avg_ts_recon_loss,
        'pocket_loss': avg_ts_pocket_loss,
        'aff_loss': avg_ts_aff_loss,
        'vDCC': avg_ts_vDCC,
        'vDCC_SR': avg_ts_vDCC_SR,
        'DVO': avg_ts_DVO,
        'F1': avg_ts_f1,
        'DCC_nan_count': ts_DCC_nan_count,
        'PCC': ts_PCC,
        'RMSE': ts_RMSE,
        'MAE': ts_MAE
    }
    
    if return_preds:
        return metrics, result_dict

    return metrics

def main():
    args = parse_arguments()
    experiment_name = os.path.basename(args.ckpt).replace(".pt", "")
    
    with open(args.ckpt_config, 'r') as fp:
        ckpt_config = json.load(fp)
    
    train_config = ckpt_config['train_config']
    train_config = dict_to_args(train_config) # . 으로 접근 가능한 namespace로 변환
    data_config = ckpt_config['data_config']
    
    mdl = MasterDataLoader(data_config, train_config)
    ts_dataloader = mdl.get_ts_dataloader()
    
    print("Building model...")
    
    model = CustomSwinUnetr(out_channels=1)
    
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    print(f"Successfully load ckpt...")
    print("Starting evaluation...")
    
    metrics = eval_model(model, ts_dataloader, train_config, data_config, experiment_name, args.output_dir, args.gpu)
    
if __name__ == "__main__":
    main()