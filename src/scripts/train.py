import argparse
import wandb
import json
import os, sys
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler

import warnings

warnings.filterwarnings("ignore")

import sys
from model.model import CustomSwinUnetr
sys.path.append("../../../../SCRIPTS")
from data.dataloader import MasterDataLoader
from _unspecified import (
    fix_seed,
    EarlyStopping,
    DiceWithLogitsLoss,
    rotate_3d_6faces,
    calc_batch_vDCC_count_nan_with_logit,
    calc_batch_DVO_with_logit,
    SoftDiceWithLogitsLoss,
    FocalLoss,
    CosineAnnealingWarmUpRestarts,
    add_gaussian_noise,
    parse_int_list,
    parse_float_list,
    parse_str_list,
    override_args_from_json
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument("--wandb_config", type=str, default=None, help="Path to result JSON config file after wandb sweep")
    parser.add_argument("--data_config", type=str, required=True, help='Json configuration file path specifies train/test data splits and metadata for learning.')
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help='Output directory to save best model validation results and checkpoint')
    parser.add_argument("--num_workers", type=int, default=6, help="Number of dataloader workers")
    
    # For Reproducibility
    parser.add_argument('--seed', type=int, default=312, help='Random seed for training, (-1 for random)')
    parser.add_argument("--ckpt", type=str, help="Model checkpoint path if exists")

    # =============== Model Args (sw: swin transformer encoder, mol: molecule encoder, dec: decoder, ca: cross attn) ===============
    parser.add_argument("--in_channels", type=int, default=21, help="Input channel size")
    parser.add_argument("--feature_size", type=int, default=48, help="Dimension of the first embedding token")
    parser.add_argument("--sw_depths", type=parse_int_list, default=[2, 2, 2], help="Swin Transformer layer depths. Each int means number of swin attention operations in each layer")
    parser.add_argument("--sw_window_size", type=int, default=7, help="Window size for swin attention operation")
    parser.add_argument("--sw_patch_size", type=parse_int_list, default=[2, 2, 2], help="Patch size (only used when model is SwinUnetr)")
    parser.add_argument("--sw_num_heads", type=parse_int_list, default=[3, 6, 12], help="Head numbers for multi-head attention in each Swin Transformer layer")
    parser.add_argument("--sw_mlp_ratio", type=float, default=4.0, help="MLP hidden size ratio for feedforward networks in Swin Transformer")
    parser.add_argument("--sw_qkv_bias", type=bool, default=True, help="Whether to use bias in query, key, value projections in Swin Transformer attention")
    parser.add_argument("--sw_drop_rate", type=float, default=0.0, help="Dropout rate for feedforward networks in Swin Transformer")
    parser.add_argument("--sw_attn_drop_rate", type=float, default=0.0, help="Dropout rate for swin attention operations in Swin Transformer")
    parser.add_argument("--sw_drop_path_rate", type=float, default=0.1, help="Drop path rate for stochastic depth in Swin Transformer")
    parser.add_argument("--sw_act", type=str, default="gelu", help="Activation function type for Swin Transformer")
    
    parser.add_argument("--mol_encoder_types", type=str, default=["attnfp", "schnet", "egnn"], help="Molecular graph encoder combinations (choices: attnfp, schnet, egnn, gcn, gat)")
    parser.add_argument("--mol_in_channels", type=int, default=54)
    parser.add_argument("--mol_hidden_channels", type=int, default=128)
    parser.add_argument("--mol_out_channels", type=int, default=128)
    parser.add_argument("--mol_num_layers", type=int, default=2)
    parser.add_argument("--mol_num_interactions_3d", type=int, default=3)
    parser.add_argument("--mol_dropout_3d", type=float, default=0.15)
    parser.add_argument("--mol_cutoff_3d", type=int, default=10)
    parser.add_argument("--mol_num_filters_schnet", type=int, default=128)
    parser.add_argument("--mol_edge_num_gaussian_schnet", type=int, default=50)
    parser.add_argument("--mol_edge_num_fourier_feats_egnn", type=int, default=3)
    parser.add_argument("--mol_soft_edge_egnn", type=bool, default=False)
    parser.add_argument("--mol_act", type=str, default="mish")

    parser.add_argument("--dec_drop_rate", type=float, default=0.1)
    parser.add_argument("--dec_act", type=str, default="gelu")

    parser.add_argument("--ca_num_heads", type=int, default=8)
    parser.add_argument("--ca_dropout", type=float, default=0.1)
    
    # =============== Training Args ===============
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--tr_subset_ratio", type=float, default=1.0, help="Ratio of training data to use per epoch")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer (=basal level of cosine scheduler)")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training")
    parser.add_argument("--grad_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients before performing a backward/update pass")

    # loss
    parser.add_argument("--recon_loss_weight", type=float, default=1.0,
                    help="Loss weight for reconstruction")
    parser.add_argument("--pocket_loss_weight", type=float, default=2.0,
                        help="Loss weight for pocket prediction")
    parser.add_argument("--aff_loss_weight", type=float, default=0.2,
                        help="Loss weight for affinity prediction")
    parser.add_argument("--recon_loss_type", type=str, default="mse", choices=["mse", "smooth_l1", "huber"], help="Reconstruction loss function type")                    
    parser.add_argument("--aff_loss_type", type=str, default="mse", choices=["mse", "smooth_l1", "huber"], help="Affinity prediction loss function type")
    parser.add_argument("--pocket_loss_types", type=parse_str_list, default="soft_dice bce", help="Muitiple pocket loss types to use, seperated with single space (e.g., 'soft_dice dice bce')")

    # optimizer    
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"], help="Optimizer types")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay for Adam optimizer")

    # scheduler
    parser.add_argument("--scheduler_T0", type=int, default=100, 
                        help="First cycle epoch count for CosineAnnealingWarmUpRestarts")
    parser.add_argument("--scheduler_T_mult", type=int, default=1, 
                        help="Cycle length multiplier for CosineAnnealingWarmUpRestarts")
    parser.add_argument("--scheduler_eta_max", type=float, default=1e-3, 
                        help="Maximum learning rate for CosineAnnealingWarmUpRestarts")
    parser.add_argument("--scheduler_T_up", type=int, default=10, 
                        help="Warmup epoch count for CosineAnnealingWarmUpRestarts")
    parser.add_argument("--scheduler_gamma", type=float, default=0.7,
                        help="Decay factor per cycle for CosineAnnealingWarmUpRestarts")

    # augmentation
    parser.add_argument("--rotation_prob", type=float, default=0.3, help="Probability of applying random rotation")
    parser.add_argument("--label_noise_std", type=float, default=0.15, help="Std value for label noise injection")

    # earlystop
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")

    # =============== Evaluation Args ===============
    parser.add_argument("--DCC_threshold", type=float, default=0.5, help="Threshold to classify voxel as pocket (DCC)")
    parser.add_argument("--DVO_threshold", type=float, default=0.5, help="Threshold to classify voxel as pocket (DVO)")
    parser.add_argument("--DCC_SR_threshold", type=float, default=4.0, help="Max DCC value for success rate calculation in validation")

    args = parser.parse_args()
    args = override_args_from_json(args, args.wandb_config)
    return args


def train_model(
    model, tr_dataloader, vl_dataloader, train_config, data_config, experiment_name
):

    wandb.init(
        project=f"InSiteDTA",
        name=experiment_name,
        config=train_config,
    )
    wandb.watch(model, log="gradients", log_freq=50)

    aug_generator = torch.Generator()
    if train_config.seed != -1:
        aug_generator.manual_seed(train_config.seed)

    device = torch.device(f"cuda:{train_config.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    optimizer = None
    if train_config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    elif train_config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
    elif train_config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            momentum=train_config.sgd_momentum
        )
    
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=train_config.scheduler_T0,  # 첫 번째 사이클 에폭 수
        T_mult=train_config.scheduler_T_mult,  # 2면 다음 사이클들: 30 → 60 → 120 에포크
        eta_max=train_config.scheduler_eta_max,  # 최대 학습률 (기존 ReduceLROnPlateau보다 높게)
        T_up=train_config.scheduler_T_up,  # 5 에포크 warmup
        gamma=train_config.scheduler_gamma,  # 사이클마다 최대 lr을 0.7배로 감소
    )

    early_stopping = EarlyStopping(patience=train_config.patience)
    
    scaler = GradScaler()

    # Reconstruction loss
    if train_config.recon_loss_type == "mse":
        recon_criterion = nn.MSELoss()
    elif train_config.recon_loss_type == "huber":
        recon_criterion = nn.HuberLoss()
    elif train_config.recon_loss_type == "smooth_l1":
        recon_criterion = nn.SmoothL1Loss()

    # Affinity loss
    if train_config.aff_loss_type == "mse":
        aff_criterion = nn.MSELoss()
    elif train_config.aff_loss_type == "huber":
        aff_criterion = nn.HuberLoss()
    elif train_config.aff_loss_type == "smooth_l1":
        aff_criterion = nn.SmoothL1Loss()

    # Pocket loss
    pocket_loss_functions = {}
    if "dice" in train_config.pocket_loss_types:
        pocket_loss_functions["dice"] = DiceWithLogitsLoss()
    if "soft_dice" in train_config.pocket_loss_types:
        pocket_loss_functions["soft_dice"] = SoftDiceWithLogitsLoss()
    if "bce" in train_config.pocket_loss_types:
        pocket_loss_functions["bce"] = nn.BCEWithLogitsLoss()

    # Load loss weight
    recon_weight = train_config.recon_loss_weight
    pocket_weight = train_config.pocket_loss_weight
    aff_weight = train_config.aff_loss_weight

    best_vl_total_loss = float("inf")
    best_model = None
    nan_count_per_epoch = []

    tr_dataset_size = len(tr_dataloader.dataset)
    vl_dataset_size = len(vl_dataloader.dataset)

    tr_steps_per_epoch = len(tr_dataloader)

    if train_config.tr_subset_ratio < 1.0:
        tr_iter_limit = int(tr_steps_per_epoch * train_config.tr_subset_ratio)
        tr_iter_limit = max(1, tr_iter_limit)
        print(f"Speed-up: Using {train_config.tr_subset_ratio*100}% data (Train: {tr_iter_limit}/{tr_steps_per_epoch}, Validation on full set")
    else:
        tr_iter_limit = tr_steps_per_epoch

    # TRAINING
    for epoch in range(train_config.epochs):
        # Training phase
        model.train()
        tr_recon_losses = 0
        tr_pocket_losses = 0
        tr_aff_losses = 0
        tr_total_losses = 0
        accumulation_step = 0

        for i, sample in enumerate(tqdm(tr_dataloader)):
            if i >= tr_iter_limit:
                break
            voxel, pocket, ligand_data, b_aff = (
                sample["voxel"],
                sample["pocket_label"],
                sample["ligand_data"],
                sample["b_aff"],
            )
            if pocket.sum() == 0.0:
                print(sample['data_key'], "has no pocket.")
            voxel, pocket, ligand_data, b_aff = (
                voxel.to(device),
                pocket.to(device),
                ligand_data.to(device),
                b_aff.to(device),
            )
            
            # On-line data augmentation
            if (
                torch.rand(1, generator=aug_generator).item()
                < train_config.rotation_prob
            ):
                voxel, pocket = rotate_3d_6faces(voxel, pocket)

            b_aff = add_gaussian_noise(b_aff, noise_std=train_config.label_noise_std)

            # Zero gradients only at the beginning of accumulation
            if accumulation_step == 0:
                optimizer.zero_grad()

            with autocast(device_type=device.type):
                output_voxel, output_aff = model(voxel, ligand_data)
                recon_voxel = output_voxel[:, :train_config.in_channels, ...]
                pred_pocket = output_voxel[:, -1:, ...]  # It's a logit.

                # Reconstruction loss for first 11 channels
                batch_size = voxel.size(0)
                recon_loss = torch.tensor([0]).to(device)
                # recon_loss *= recon_weight

                pocket_loss = 0
                for loss_type, loss_fn in pocket_loss_functions.items():
                    pocket_loss += loss_fn(pred_pocket, pocket)

                # Binding affinity loss
                has_valid_aff = ~torch.isnan(b_aff).all()
                if not has_valid_aff:
                    breakpoint()
                aff_loss = aff_criterion(output_aff, b_aff) if has_valid_aff else torch.tensor(0.0, device=voxel.device)
                # Total loss
                total_loss = (
                    recon_weight * recon_loss
                    + pocket_weight * pocket_loss
                    + aff_weight * aff_loss
                )
                
                # Normalize loss for gradient accumulation
                total_loss = total_loss / train_config.grad_accumulation_steps

            scaler.scale(total_loss).backward()
            accumulation_step += 1
            
            # Perform optimizer step only when accumulation is complete or at the end of epoch
            if accumulation_step == train_config.grad_accumulation_steps or i == len(tr_dataloader) - 1:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                accumulation_step = 0  # Reset accumulation counter

            # Store losses (multiply total_loss back by accumulation steps for proper logging)
            tr_recon_losses += recon_loss.detach().item()
            tr_pocket_losses += pocket_loss.detach().item()
            tr_aff_losses += aff_loss.detach().item()
            tr_total_losses += total_loss.detach().item() * train_config.grad_accumulation_steps

            # del voxel, pocket, ligand_data, b_aff
            # del output_voxel, output_aff, recon_voxel, pred_pocket
            # del recon_loss, pocket_dice_loss, pocket_bce_loss, pocket_loss, aff_loss, total_loss

            if i % 20 == 0:
                torch.cuda.empty_cache()

        avg_tr_recon_loss = tr_recon_losses / tr_iter_limit
        avg_tr_pocket_loss = tr_pocket_losses / tr_iter_limit
        avg_tr_aff_loss = tr_aff_losses / tr_iter_limit
        avg_tr_total_loss = tr_total_losses / tr_iter_limit

        # Validation phase
        model.eval()
        vl_recon_losses = []
        vl_pocket_losses = []
        vl_aff_losses = []
        vl_total_losses = []
        vl_vDCCs = []  # voxel-base calculated DCC (not per-atomtype)
        vl_DVOs = []
        vl_nan_indice = []
        vl_aff_all_preds = []
        vl_aff_all_targets = []

        # VALIDATION
        with torch.no_grad():
            for sample in tqdm(vl_dataloader):
                voxel, pocket, ligand_data, b_aff = (
                    sample["voxel"],
                    sample["pocket_label"],
                    sample["ligand_data"],
                    sample["b_aff"],
                )
                voxel, pocket, ligand_data, b_aff = (
                    voxel.to(device),
                    pocket.to(device),
                    ligand_data.to(device),
                    b_aff.to(device),
                )
                
                if pocket.sum() == 0.0:
                    print(sample['data_key'])

                with autocast(device_type=device.type):
                    output_voxel, output_aff = model(voxel, ligand_data)
                    recon_voxel = output_voxel[:, :train_config.in_channels, ...]
                    pred_pocket = output_voxel[:, -1:, ...]

                    batch_size = voxel.size(0)
                    recon_loss = torch.tensor([0]).to(device)

                    pocket_loss = 0
                    for loss_type, loss_fn in pocket_loss_functions.items():
                        pocket_loss += loss_fn(pred_pocket, pocket) * batch_size
                    
                    has_valid_aff = ~torch.isnan(b_aff).all()
                    aff_loss = aff_criterion(output_aff, b_aff) * batch_size if has_valid_aff else torch.tensor(0.0, device=voxel.device)

                    total_loss = (
                        recon_weight * recon_loss
                        + pocket_weight * pocket_loss
                        + aff_weight * aff_loss
                    )

                # val metrics 계산
                vDCC, nan_index = calc_batch_vDCC_count_nan_with_logit(
                    pred_pocket,
                    pocket,
                    voxel_size=data_config["voxel_size"],
                    threshold=train_config.DCC_threshold,
                )
                    
                DVO = calc_batch_DVO_with_logit(
                    pred_pocket, pocket, threshold=train_config.DVO_threshold
                )

                vl_recon_losses.append(recon_loss.item())
                vl_pocket_losses.append(pocket_loss.item())
                vl_aff_losses.append(aff_loss.item())
                vl_total_losses.append(total_loss.item())
                vl_vDCCs += vDCC.tolist()
                vl_DVOs += DVO.tolist()
                vl_nan_indice += nan_index
                vl_aff_all_preds.append(output_aff.detach().cpu())
                vl_aff_all_targets.append(b_aff.cpu())

        avg_vl_recon_loss = sum(vl_recon_losses) / vl_dataset_size
        avg_vl_pocket_loss = sum(vl_pocket_losses) / vl_dataset_size
        avg_vl_aff_loss = sum(vl_aff_losses) / vl_dataset_size
        avg_vl_total_loss = sum(vl_total_losses) / vl_dataset_size
        avg_vl_vDCC = sum(vl_vDCCs) / vl_dataset_size
        avg_vl_vDCC_SR = (
            len([DCC for DCC in vl_vDCCs if DCC <= train_config.DCC_SR_threshold])
            / vl_dataset_size
        )
        avg_vl_DVO = sum(vl_DVOs) / vl_dataset_size
        vl_DCC_nan_count = len(vl_nan_indice)
        nan_count_per_epoch.append(vl_DCC_nan_count)

        epoch_preds = torch.cat(vl_aff_all_preds, dim=0).cpu().numpy().squeeze()
        epoch_targets = torch.cat(vl_aff_all_targets, dim=0).cpu().numpy().squeeze()
        vl_PCC = np.corrcoef(epoch_preds, epoch_targets)[0, 1]

        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{train_config.epochs}]  "
            f"Train Loss (recon/pocket/b_aff): {avg_tr_total_loss:.4f} ({avg_tr_recon_loss:.4f}/{avg_tr_pocket_loss:.4f}/{avg_tr_aff_loss:.4f})  "
            f"Valid Loss (recon/pocket/b_aff): {avg_vl_total_loss:.4f} ({avg_vl_recon_loss:.4f}/{avg_vl_pocket_loss:.4f}/{avg_vl_aff_loss:.4f})  "
            f"Valid vDCC_{train_config.DCC_threshold}: {avg_vl_vDCC:.4f}  "
            f"Valid vDCC_{train_config.DCC_threshold}_SR_{train_config.DCC_SR_threshold}: {avg_vl_vDCC_SR:.4f}  "
            f"Valid DVO_{train_config.DVO_threshold}: {avg_vl_DVO:.4f}  "
            f"Valid DCC_nan_count: {vl_DCC_nan_count:.4f}  "
            f"Valid PCC: {vl_PCC:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss/train/total": avg_tr_total_loss,
                "loss/train/reconstruction": avg_tr_recon_loss,
                "loss/train/pocket": avg_tr_pocket_loss,
                "loss/train/affinity": avg_tr_aff_loss,
                "loss/valid/total": avg_vl_total_loss,
                "loss/valid/reconstruction": avg_vl_recon_loss,
                "loss/valid/pocket": avg_vl_pocket_loss,
                "loss/valid/affinity": avg_vl_aff_loss,
                f"metrics/valid/vDCC_theta{train_config.DCC_threshold}": avg_vl_vDCC,
                f"metrics/valid/vDCC_theta{train_config.DCC_threshold}_SR_{train_config.DCC_SR_threshold}": avg_vl_vDCC_SR,
                f"metrics/valid/DVO_theta{train_config.DVO_threshold}": avg_vl_DVO,
                "metrics/valid/DCC_nan_count": vl_DCC_nan_count,
                "metrics/valid/PCC": vl_PCC,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        # scheduler.step(avg_vl_total_loss)
        scheduler.step()

        # Save best model
        if avg_vl_total_loss < best_vl_total_loss:
            best_vl_total_loss = avg_vl_total_loss
            best_model = model.state_dict().copy()
            # best metric 기록
            best_epoch = epoch + 1
            best_vl_total_loss = avg_vl_total_loss
            best_vl_recon_loss = avg_vl_recon_loss
            best_vl_pocket_loss = avg_vl_pocket_loss
            best_vl_aff_loss = avg_vl_aff_loss
            best_vl_vDCC = avg_vl_vDCC
            best_vl_vDCC_SR = avg_vl_vDCC_SR
            best_vl_DVO = avg_vl_DVO
            best_DCC_nan_count = vl_DCC_nan_count
            best_vl_PCC = vl_PCC
            print("Best model has saved!")

            # 최상의 모델을 W&B에 저장
            # model_artifact = wandb.Artifact(
            #     name=f"best_model_run_{wandb.run.id}",
            #     type="model",
            #     description=f"Best model from epoch {best_epoch}",
            # )
            os.makedirs(train_config.output_dir, exist_ok=True)
            torch.save(
                best_model,
                os.path.join(train_config.output_dir, f"{experiment_name}.pt"),
            )
            # model_artifact.add_file(
            #     os.path.join(train_config.output_dir, f"{experiment_name}.pt")
            # )
            # wandb.log_artifact(model_artifact)

        # Early stopping check
        end_epoch = train_config.epochs
        improved = early_stopping(avg_vl_total_loss, epoch + 1)
        if early_stopping.early_stop:
            end_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best model was at epoch {early_stopping.best_epoch}")
            break

    # Load best model
    model.load_state_dict(best_model)

    # writer 닫기
    if wandb.run is not None:
        wandb.run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_total_loss": best_vl_total_loss,
                "best_recon_loss": best_vl_recon_loss,
                "best_pocket_loss": best_vl_pocket_loss,
                "best_aff_loss": best_vl_aff_loss,
                "best_vDCC": best_vl_vDCC,
                "best_vDCC_SR": best_vl_vDCC_SR,
                "best_DVO": best_vl_DVO,
                "best_DCC_nan_count": best_DCC_nan_count,
                "best_PCC": best_vl_PCC,
                "avg_nan_count": sum(nan_count_per_epoch) / len(nan_count_per_epoch),
            }
        )

    wandb.finish()

    avg_nan_count = sum(nan_count_per_epoch) / len(nan_count_per_epoch)

    num_metrics = [
        best_vl_total_loss,
        best_vl_recon_loss,  # 숫자로 구성된 metrics들
        best_vl_pocket_loss,
        best_vl_aff_loss,
        best_vl_vDCC,
        best_vl_vDCC_SR,
        best_vl_DVO,
        best_DCC_nan_count,
        best_vl_PCC,
        avg_nan_count,
    ]

    return_metrics = [f"{best_epoch}/{end_epoch}"] + [
        round(num, 4) for num in num_metrics
    ]

    return model, return_metrics


def main():
    train_config = parse_arguments()
    
    # --- [디버깅용 코드 시작] ---
    print("\n" + "="*30)
    print("📢 [Current Local Config Check]")
    print("="*30)
    import pprint
    pprint.pprint(vars(train_config)) # 현재 로컬 코드에 적용된 모든 설정값 출력
    print("="*30 + "\n")
    # --- [디버깅용 코드 끝] ---
    
    with open(train_config.data_config, "r") as fp:
        data_config = json.load(fp)

    # get experiment name
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    experiment_name = (
        f"{timestamp}_{os.path.basename(train_config.data_config).replace('.json', '')}"
    )
    
    if train_config.seed != -1:
        fix_seed(train_config.seed)

    # Load Dataloader
    mdl = MasterDataLoader(data_config, train_config)
    tr_dataloader, vl_dataloader = mdl.get_tr_vl_dataloader()

    print("Building model...")
    model = CustomSwinUnetr(
        spatial_dims=3,
        in_channels=train_config.in_channels,
        out_channels=1,
        feature_size=train_config.feature_size,
        sw_depths=train_config.sw_depths,
        sw_window_size=train_config.sw_window_size,
        sw_patch_size=train_config.sw_patch_size,
        sw_num_heads=train_config.sw_num_heads,
        sw_mlp_ratio=train_config.sw_mlp_ratio,
        sw_qkv_bias=train_config.sw_qkv_bias,
        sw_drop_rate=train_config.sw_drop_rate,
        sw_attn_drop_rate=train_config.sw_attn_drop_rate,
        sw_drop_path_rate=train_config.sw_drop_path_rate,
        sw_act=train_config.sw_act,
        mol_encoder_types=train_config.mol_encoder_types,
        mol_in_channels=train_config.mol_in_channels,
        mol_hidden_channels=train_config.mol_hidden_channels,
        mol_out_channels=train_config.mol_out_channels,
        mol_num_layers=train_config.mol_num_layers,
        mol_num_interactions_3d=train_config.mol_num_interactions_3d,
        mol_dropout_3d=train_config.mol_dropout_3d,
        mol_cutoff_3d=train_config.mol_cutoff_3d,
        mol_num_filters_schnet=train_config.mol_num_filters_schnet,
        mol_edge_num_gaussian_schnet=train_config.mol_edge_num_gaussian_schnet,
        mol_edge_num_fourier_feats_egnn=train_config.mol_edge_num_fourier_feats_egnn,
        mol_soft_edge_egnn=train_config.mol_soft_edge_egnn,
        mol_act=train_config.mol_act,
        dec_drop_rate=train_config.dec_drop_rate,
        dec_act=train_config.dec_act,
        ca_num_heads=train_config.ca_num_heads,
        ca_dropout=train_config.ca_dropout
    )

    if train_config.ckpt:
        print(f"Load ckpt...{os.path.basename(train_config.ckpt)}")
        model.load_state_dict(torch.load(train_config.ckpt, weights_only=True))

    # 모델 학습
    print("Starting training...")
    model, metrics = train_model(
        model, tr_dataloader, vl_dataloader, train_config, data_config, experiment_name
    )

    # 설정 및 결과 저장
    config_summary = {
        "train_config": vars(train_config),
        "data_config": data_config,
        "metrics": {
            "epochs": metrics[0],
            "best_vl_total_loss": metrics[1],
            "best_vl_recon_loss": metrics[2],
            "best_vl_pocket_loss": metrics[3],
            "best_vl_aff_loss": metrics[4],
            "best_vl_vDCC": metrics[5],
            "best_vl_vDCC_SR": metrics[6],
            "best_vl_DVO": metrics[7],
            "best_DCC_nan_count": metrics[8],
            "best_vl_PCC": metrics[9],
            "avg_nan_count": metrics[10],
        },
    }

    # 결과 JSON으로 저장
    with open(
        os.path.join(train_config.output_dir, f"{experiment_name}_results.json"), "w"
    ) as fp:
        json.dump(config_summary, fp, indent=4)

    print(f"Training completed. Results:")
    print(f"  Best epoch / Total epochs: {metrics[0]}")
    print(f"  Best validation total loss: {metrics[1]:.4f}")
    print(f"  Best validation PCC: {metrics[9]:.4f}")
    print(f"  Best validation vDCC SR: {metrics[6]:.4f}")

    return experiment_name


if __name__ == "__main__":
    main()