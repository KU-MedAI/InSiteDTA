# TODO: 저장을 ckpt 폴더에 맞게 해서 inference 폴더 또는 reproduce 폴더와 일치하게.
"""
data directory, hyperparams -> training cfg file
training cfg file -> training/validation -> ckpt saving
evaluation by inference code? options 조절이 안되는데 out channels 만 조절하면 되려나
"""
import argparse
import copy
import math
import os
import sys
import wandb
import json
import numpy as np

from functools import partial
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
_REPO_ROOT = os.path.abspath(os.path.join(_MODEL_DIR, "../../../"))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _MODEL_DIR)

from src.scripts.dataloader import MasterDataLoader
from model import InSiteDTA
from src.scripts.utils import print_args, parse_to_list
from src.scripts.utils_train import (
    fix_seed,
    CosineAnnealingWarmUpRestarts,
    rotate_3d_6faces,
    add_gaussian_noise,
    EarlyStopping,
    CosineAnnealingWarmUpRestarts,
    SoftDiceWithLogitsLoss,
    calc_DCC_with_logit,
    calc_DVO_with_logit,
)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="GPU device ID to use")
    parser.add_argument(
        "--wandb_config",
        type=str,
        help="Path to result JSON config file after wandb sweep",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default=os.path.join(_REPO_ROOT, "src/data/datasplit_preset/data_config_CleanSplit0.json"),
        help="Json configuration file path specifies train/test data splits and metadata for learning.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "../../ckpts/02_improve_training/1a_distill_pocket_soft"),
        help="Output directory to save best model validation results and checkpoint",
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of dataloader workers"
    )

    # For Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=312,
        help="Random seed for training, (-1 for random)",
    )

    # =============== Model Args (sw: swin transformer encoder, mol: molecule encoder, dec: decoder, ca: cross attention module) ===============
    parser.add_argument(
        "--in_channels", type=int, default=21, help="Input channel size"
    )
    parser.add_argument(
        "--out_channels", type=int, default=1, help="Output channel size"
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=48,
        help="Dimension of the first embedding token",
    )

    parser.add_argument(
        "--sw_depths",
        type=partial(parse_to_list, type=int),
        default=[2, 2, 2],
        help="Swin Transformer layer depths. Each int means number of swin attention operations in each layer",
    )
    parser.add_argument(
        "--sw_window_size",
        type=int,
        default=7,
        help="Window size for swin attention operation",
    )
    parser.add_argument(
        "--sw_patch_size",
        type=partial(parse_to_list, type=int),
        default=[2, 2, 2],
        help="Patch size (only used when model is SwinUnetr)",
    )
    parser.add_argument(
        "--sw_num_heads",
        type=partial(parse_to_list, type=int),
        default=[3, 6, 12],
        help="Head numbers for multi-head attention in each Swin Transformer layer",
    )
    parser.add_argument(
        "--sw_mlp_ratio",
        type=float,
        default=4.0,
        help="MLP hidden size ratio for feedforward networks in Swin Transformer",
    )
    parser.add_argument(
        "--sw_qkv_bias",
        type=bool,
        default=True,
        help="Whether to use bias in query, key, value projections in Swin Transformer attention",
    )
    parser.add_argument(
        "--sw_drop_rate",
        type=float,
        default=0.0,
        help="Dropout rate for feedforward networks in Swin Transformer",
    )
    parser.add_argument(
        "--sw_attn_drop_rate",
        type=float,
        default=0.0,
        help="Dropout rate for swin attention operations in Swin Transformer",
    )
    parser.add_argument(
        "--sw_drop_path_rate",
        type=float,
        default=0.1,
        help="Drop path rate for stochastic depth in Swin Transformer",
    )
    parser.add_argument(
        "--sw_act",
        type=str,
        default="gelu",
        help="Activation function type for Swin Transformer",
    )

    parser.add_argument(
        "--mol_encoder_types",
        type=str,
        default=["attnfp", "schnet", "egnn"],
        help="Molecular graph encoder combinations (choices: attnfp, schnet, egnn, gcn, gat)",
    )
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
    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of training epochs"
    )
    parser.add_argument(
        "--tr_subset_ratio",
        type=float,
        default=1.0,
        help="Ratio of training data to use per epoch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer (=basal level of cosine scheduler)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=48, help="Batch size for training"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients before performing a backward/update pass",
    )

    # loss configs
    parser.add_argument(
        "--poc_loss_weight",
        type=float,
        default=2.0,
        help="Loss weight for pocket prediction",
    )
    parser.add_argument(
        "--aff_loss_weight",
        type=float,
        default=0.2,
        help="Loss weight for affinity prediction",
    )

    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam"],
        help="Optimizer types",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for Adam(w) optimizer",
    )

    # scheduler
    parser.add_argument(
        "--scheduler_T0",
        type=int,
        default=100,
        help="First cycle epoch count for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_T_mult",
        type=int,
        default=1,
        help="Cycle length multiplier for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_eta_max",
        type=float,
        default=1e-3,
        help="Maximum learning rate for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_T_up",
        type=int,
        default=10,
        help="Warmup epoch count for CosineAnnealingWarmUpRestarts",
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.7,
        help="Decay factor per cycle for CosineAnnealingWarmUpRestarts",
    )

    # augmentation
    parser.add_argument(
        "--rotation_prob",
        type=float,
        default=0.3,
        help="Probability of applying random rotation",
    )
    parser.add_argument(
        "--label_noise_std",
        type=float,
        default=0.15,
        help="Std value for label noise injection",
    )
    parser.add_argument(
        "--channel_dropout_prob",
        type=float,
        default=0.1,
        help="Probability of dropping each voxel channel (set to 0 to disable)",
    )
    parser.add_argument(
        "--voxel_noise_std",
        type=float,
        default=0.05,
        help="Std of Gaussian noise added to voxel input (set to 0 to disable)",
    )

    # distillation
    parser.add_argument(
        "--ema_momentum_start",
        type=float,
        default=0.996,
        help="EMA momentum at the start of training (cosine schedule)",
    )
    parser.add_argument(
        "--ema_momentum_end",
        type=float,
        default=1.0,
        help="EMA momentum at the end of training (cosine schedule)",
    )
    parser.add_argument(
        "--distill_weight",
        type=float,
        default=1.0,
        help="Weight for distillation loss (α)",
    )
    parser.add_argument(
        "--distill_tau",
        type=float,
        default=1.0,
        help="Temperature for pocket soft label (τ)",
    )

    # earlystop
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0,
        help="Minimum improvement in validation loss to reset early stopping counter and save best model",
    )

    # =============== Evaluation Args ===============
    parser.add_argument(
        "--DCC_threshold",
        type=float,
        default=0.5,
        help="Threshold to classify voxel as pocket (DCC)",
    )
    parser.add_argument(
        "--DVO_threshold",
        type=float,
        default=0.5,
        help="Threshold to classify voxel as pocket (DVO)",
    )
    parser.add_argument(
        "--DCC_SR_threshold",
        type=float,
        default=4.0,
        help="Max DCC value for success rate calculation in validation",
    )

    args = parser.parse_args()
    return args


def get_configs() -> tuple[dict, dict]:
    train_cfg = get_arguments()
    print_args(train_cfg)

    with open(train_cfg.data_config, "r") as fp:
        data_cfg = json.load(fp)

    return data_cfg, vars(train_cfg)


def create_ema_teacher(model):
    """Create EMA teacher as a deep copy with frozen gradients."""
    teacher = copy.deepcopy(model)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.no_grad()
def update_ema_teacher(teacher, student, momentum):
    """Update teacher parameters with EMA."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)


def get_ema_momentum(epoch, total_epochs, m_start, m_end):
    """Cosine schedule for EMA momentum: m_start → m_end."""
    return m_end - (m_end - m_start) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


def train_model(
    model: nn.Module,
    tr_loader: torch.utils.data.DataLoader,
    vl_loader: torch.utils.data.DataLoader,
    train_cfg: dict,
    voxel_size: int,
    exp_name: str,
):
    print("Start training!\n")
    os.makedirs(train_cfg["save_dir"], exist_ok=True)

    wandb.init(
        project="InSiteDTA-v2",
        name=exp_name,
        config=train_cfg,
    )
    wandb.watch(model, log="gradients", log_freq=50)

    aug_generator = torch.Generator()
    if train_cfg["seed"] != -1:
        aug_generator.manual_seed(train_cfg["seed"])

    device = torch.device(
        f"cuda:{train_cfg['device']}" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    # EMA teacher
    teacher = create_ema_teacher(model)
    teacher.to(device)
    teacher.eval()
    distill_weight = train_cfg["distill_weight"]
    distill_tau = train_cfg["distill_tau"]

    early_stopping = EarlyStopping(
        patience=train_cfg["patience"], min_delta=train_cfg["min_delta"]
    )
    scaler = GradScaler()

    if train_cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
    elif train_cfg["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer,
        T_0=train_cfg["scheduler_T0"],
        T_mult=train_cfg["scheduler_T_mult"],
        eta_max=train_cfg["scheduler_eta_max"],
        T_up=train_cfg["scheduler_T_up"],
        gamma=train_cfg["scheduler_gamma"],
    )

    aff_criterion = nn.MSELoss()
    poc_criterion_bce = nn.BCEWithLogitsLoss()
    poc_criterion_softdice = SoftDiceWithLogitsLoss()

    poc_weight = train_cfg["poc_loss_weight"]
    aff_weight = train_cfg["aff_loss_weight"]

    best_vl_PCC = -float("inf")
    best_vl_loss = float("inf")
    best_model = None
    best_teacher = None
    nan_count_ls = []

    vl_dataset_size = len(vl_loader.dataset)

    # Speed-up training by using a data subset
    tr_steps = len(tr_loader)
    if train_cfg["tr_subset_ratio"] < 1.0:
        tr_steps_limit = int(tr_steps * train_cfg["tr_subset_ratio"])
        tr_steps_limit = max(1, tr_steps_limit)
        print(
            f"Speed-up: Using {train_cfg['tr_subset_ratio']*100}% data (Train: {tr_steps_limit}/{tr_steps}, Validation on full set"
        )
    else:
        tr_steps_limit = tr_steps

    for epoch in range(train_cfg["epochs"]):
        # EMA momentum for this epoch
        ema_momentum = get_ema_momentum(
            epoch, train_cfg["epochs"],
            train_cfg["ema_momentum_start"], train_cfg["ema_momentum_end"]
        )

        # [TRAINING]
        model.train()
        tr_poc_loss = 0
        tr_aff_loss = 0
        tr_distill_loss = 0
        tr_total_loss = 0
        accumulation_step = 0

        for i, sample in enumerate(tqdm(tr_loader)):
            if i >= tr_steps_limit:
                break

            voxel, pocket, lig_data, true_aff = (
                sample["voxel"],
                sample["pocket_label"],
                sample["lig_data"],
                sample["true_aff"],
            )

            if pocket.sum() == 0.0:
                print(sample["data_key"], "has no pocket.")

            voxel, pocket, lig_data, true_aff = (
                voxel.to(device),
                pocket.to(device),
                lig_data.to(device),
                true_aff.to(device),
            )

            # On-line data augmentation: rotation (shared between teacher and student)
            if (
                torch.rand(1, generator=aug_generator).item()
                < train_cfg["rotation_prob"]
            ):
                voxel, pocket = rotate_3d_6faces(voxel, pocket)

            # Teacher forward (clean input, same rotation)
            with torch.no_grad():
                with autocast(device_type=device.type):
                    teacher_poc, _ = teacher(voxel, lig_data)

            # Student augmentation (teacher does not see these)
            if train_cfg["channel_dropout_prob"] > 0:
                ch_mask = torch.rand(voxel.shape[1], device=device) > train_cfg["channel_dropout_prob"]
                voxel = voxel * ch_mask[None, :, None, None, None]

            if train_cfg["voxel_noise_std"] > 0:
                voxel = voxel + torch.randn_like(voxel) * train_cfg["voxel_noise_std"]

            true_aff = add_gaussian_noise(
                true_aff, noise_std=train_cfg["label_noise_std"]
            )

            if accumulation_step == 0:
                optimizer.zero_grad()

            with autocast(device_type=device.type):
                pred_poc, pred_aff = model(voxel, lig_data)

                # GT losses
                has_aff_labels = ~torch.isnan(true_aff).all()
                aff_loss = (
                    aff_criterion(pred_aff, true_aff)
                    if has_aff_labels
                    else torch.tensor(0.0, device=voxel.device)
                )
                poc_loss = poc_criterion_bce(pred_poc, pocket) + poc_criterion_softdice(
                    pred_poc, pocket
                )

                # Distillation loss: BCE with teacher soft label
                teacher_soft_label = torch.sigmoid(teacher_poc.detach() / distill_tau)
                distill_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_poc, teacher_soft_label
                )

                total_loss = (
                    aff_weight * aff_loss
                    + poc_weight * poc_loss
                    + distill_weight * distill_loss
                )

                # Normalize loss for gradient accumulation
                total_loss = total_loss / train_cfg["grad_accumulation_steps"]

            scaler.scale(total_loss).backward()
            accumulation_step += 1

            # Perform optimizer step only when accumulation is complete or at the end of epoch
            if (
                accumulation_step == train_cfg["grad_accumulation_steps"]
                or i == tr_steps_limit - 1
            ):
                scaler.step(optimizer)
                scaler.update()
                update_ema_teacher(teacher, model, ema_momentum)
                accumulation_step = 0

            # Store loss
            tr_poc_loss += poc_loss.detach().item()
            tr_aff_loss += aff_loss.detach().item()
            tr_distill_loss += distill_loss.detach().item()
            tr_total_loss += (
                total_loss.detach().item() * train_cfg["grad_accumulation_steps"]
            )

        avg_tr_poc_loss = tr_poc_loss / tr_steps_limit
        avg_tr_aff_loss = tr_aff_loss / tr_steps_limit
        avg_tr_distill_loss = tr_distill_loss / tr_steps_limit
        avg_tr_total_loss = tr_total_loss / tr_steps_limit

        # [VALIDATION]
        model.eval()
        vl_poc_loss = []
        vl_aff_loss = []
        vl_total_loss = []
        vl_DCCs = []
        vl_DVOs = []
        vl_nan_indices = []
        vl_pred_aff_values = []
        vl_true_aff_values = []

        # Teacher validation metrics
        teacher_vl_pred_aff_values = []
        teacher_vl_aff_loss_sum = 0.0
        teacher_vl_DCCs = []

        with torch.no_grad():
            for sample in tqdm(vl_loader):
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

                batch_size = voxel.size(0)

                with autocast(device_type=device.type):
                    pred_poc, pred_aff = model(voxel, lig_data)
                    teacher_poc, teacher_aff = teacher(voxel, lig_data)

                    poc_bce_loss = poc_criterion_bce(pred_poc, pocket) * batch_size
                    poc_softdice_loss = poc_criterion_softdice(pred_poc, pocket) * batch_size
                    poc_loss = poc_bce_loss + poc_softdice_loss

                    has_aff_labels = ~torch.isnan(true_aff).all()
                    aff_loss = (
                        aff_criterion(pred_aff, true_aff) * batch_size
                        if has_aff_labels
                        else torch.tensor(0.0, device=voxel.device)
                    )
                    teacher_aff_loss = (
                        aff_criterion(teacher_aff, true_aff) * batch_size
                        if has_aff_labels
                        else torch.tensor(0.0, device=voxel.device)
                    )

                    total_loss = poc_weight * poc_loss + aff_weight * aff_loss

                # Calc validation metrics
                DCC, nan_index = calc_DCC_with_logit(
                    pred_poc,
                    pocket,
                    voxel_size=voxel_size,
                    threshold=train_cfg["DCC_threshold"],
                )

                DVO = calc_DVO_with_logit(
                    pred_poc, pocket, threshold=train_cfg["DVO_threshold"]
                )

                # Teacher DCC
                teacher_DCC, _ = calc_DCC_with_logit(
                    teacher_poc,
                    pocket,
                    voxel_size=voxel_size,
                    threshold=train_cfg["DCC_threshold"],
                )

                vl_poc_loss.append(poc_loss.item())
                vl_aff_loss.append(aff_loss.item())
                vl_total_loss.append(total_loss.item())
                vl_DCCs += DCC.tolist()
                vl_DVOs += DVO.tolist()
                vl_nan_indices += nan_index
                vl_pred_aff_values.append(pred_aff.detach().cpu())
                vl_true_aff_values.append(true_aff.cpu())

                teacher_vl_aff_loss_sum += teacher_aff_loss.item()
                teacher_vl_pred_aff_values.append(teacher_aff.detach().cpu())
                teacher_vl_DCCs += teacher_DCC.tolist()
                ############ end of the epoch ############

        avg_vl_poc_loss = sum(vl_poc_loss) / vl_dataset_size
        avg_vl_aff_loss = sum(vl_aff_loss) / vl_dataset_size
        avg_vl_total_loss = sum(vl_total_loss) / vl_dataset_size
        avg_vl_DCC = sum(vl_DCCs) / vl_dataset_size
        avg_vl_DCC_SR = (
            len([DCC for DCC in vl_DCCs if DCC <= train_cfg["DCC_SR_threshold"]])
            / vl_dataset_size
        )
        avg_vl_DVO = sum(vl_DVOs) / vl_dataset_size
        vl_DCC_nan_count = len(vl_nan_indices)
        nan_count_ls.append(vl_DCC_nan_count)

        epoch_pred_aff_values = (
            torch.cat(vl_pred_aff_values, dim=0).cpu().numpy().squeeze()
        )
        epoch_true_aff_values = (
            torch.cat(vl_true_aff_values, dim=0).cpu().numpy().squeeze()
        )
        vl_PCC = np.corrcoef(epoch_pred_aff_values, epoch_true_aff_values)[0, 1]

        # Teacher validation metrics
        avg_teacher_vl_aff_loss = teacher_vl_aff_loss_sum / vl_dataset_size
        avg_teacher_vl_DCC_SR = (
            len([d for d in teacher_vl_DCCs if d <= train_cfg["DCC_SR_threshold"]])
            / vl_dataset_size
        )
        teacher_pred_aff_np = (
            torch.cat(teacher_vl_pred_aff_values, dim=0).cpu().numpy().squeeze()
        )
        teacher_vl_PCC = np.corrcoef(teacher_pred_aff_np, epoch_true_aff_values)[0, 1]

        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{train_cfg['epochs']}]  "
            f"Train Loss (pocket/affinity): {avg_tr_total_loss:.4f} ({avg_tr_poc_loss:.4f}/{avg_tr_aff_loss:.4f})  "
            f"Valid Loss (pocket/affinity): {avg_vl_total_loss:.4f} ({avg_vl_poc_loss:.4f}/{avg_vl_aff_loss:.4f})  "
            f"Valid DCC_{train_cfg['DCC_threshold']}: {avg_vl_DCC:.4f}  "
            f"Valid DCC_{train_cfg['DCC_threshold']}_SR_{train_cfg['DCC_SR_threshold']}: {avg_vl_DCC_SR:.4f}  "
            f"Valid DVO_{train_cfg['DVO_threshold']}: {avg_vl_DVO:.4f}  "
            f"Valid DCC_nan_count: {vl_DCC_nan_count}  "
            f"Valid PCC: {vl_PCC:.4f}  "
            f"Teacher PCC: {teacher_vl_PCC:.4f}  "
            f"Teacher DCC_SR: {avg_teacher_vl_DCC_SR:.4f}  "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        wandb.log(
            {
                "epoch": epoch + 1,
                "loss/train/total": avg_tr_total_loss,
                "loss/train/pocket": avg_tr_poc_loss,
                "loss/train/affinity": avg_tr_aff_loss,
                "loss/train/distill": avg_tr_distill_loss,
                "ema_momentum": ema_momentum,
                "loss/valid/total": avg_vl_total_loss,
                "loss/valid/pocket": avg_vl_poc_loss,
                "loss/valid/affinity": avg_vl_aff_loss,
                f"metrics/valid/DCC_theta{train_cfg['DCC_threshold']}": avg_vl_DCC,
                f"metrics/valid/DCC_theta{train_cfg['DCC_threshold']}_SR_{train_cfg['DCC_SR_threshold']}": avg_vl_DCC_SR,
                f"metrics/valid/DVO_theta{train_cfg['DVO_threshold']}": avg_vl_DVO,
                "metrics/valid/DCC_nan_count": vl_DCC_nan_count,
                "metrics/valid/PCC": vl_PCC,
                "metrics/valid/teacher_PCC": teacher_vl_PCC,
                "metrics/valid/teacher_DCC_SR": avg_teacher_vl_DCC_SR,
                "loss/valid/teacher_affinity": avg_teacher_vl_aff_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step()

        # Save best model (PCC 기준)
        if not np.isnan(vl_PCC) and vl_PCC > best_vl_PCC:
            best_vl_PCC = vl_PCC
            best_vl_loss = avg_vl_total_loss

            best_model = model.state_dict().copy()
            best_teacher = teacher.state_dict().copy()
            best_epoch = epoch + 1
            best_vl_poc_loss = avg_vl_poc_loss
            best_vl_aff_loss = avg_vl_aff_loss
            best_vl_DCC = avg_vl_DCC
            best_vl_DCC_SR = avg_vl_DCC_SR
            best_vl_DVO = avg_vl_DVO
            best_DCC_nan_count = vl_DCC_nan_count
            best_teacher_vl_PCC = teacher_vl_PCC
            best_teacher_vl_DCC_SR = avg_teacher_vl_DCC_SR
            best_teacher_vl_aff_loss = avg_teacher_vl_aff_loss
            print("Best model has saved!")

            torch.save(
                best_model,
                os.path.join(train_cfg["save_dir"], f"{exp_name}.pt"),
            )
            torch.save(
                best_teacher,
                os.path.join(train_cfg["save_dir"], f"{exp_name}_teacher.pt"),
            )

        # Early stop (PCC 기준, NaN 시 loss fallback)
        _es_metric = -vl_PCC if not np.isnan(vl_PCC) else avg_vl_total_loss
        early_stopping(_es_metric, epoch + 1)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            print(f"Best model was at epoch {early_stopping.best_epoch}")
            break

    # Close wandb writer
    if wandb.run is not None:
        wandb.run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_total_loss": best_vl_loss,
                "best_pocket_loss": best_vl_poc_loss,
                "best_aff_loss": best_vl_aff_loss,
                "best_DCC": best_vl_DCC,
                "best_DCC_SR": best_vl_DCC_SR,
                "best_DVO": best_vl_DVO,
                "best_DCC_nan_count": best_DCC_nan_count,
                "best_PCC": best_vl_PCC,
                "best_teacher_PCC": best_teacher_vl_PCC,
                "best_teacher_DCC_SR": best_teacher_vl_DCC_SR,
                "best_teacher_aff_loss": best_teacher_vl_aff_loss,
                "avg_nan_count": sum(nan_count_ls) / len(nan_count_ls),
            }
        )

    wandb.finish()

    avg_nan_count = sum(nan_count_ls) / len(nan_count_ls)

    num_metrics = [
        best_vl_loss,
        best_vl_poc_loss,
        best_vl_aff_loss,
        best_vl_DCC,
        best_vl_DCC_SR,
        best_vl_DVO,
        best_DCC_nan_count,
        best_vl_PCC,
        avg_nan_count,
        best_teacher_vl_PCC,
        best_teacher_vl_DCC_SR,
        best_teacher_vl_aff_loss,
    ]

    return_metrics = [f"{best_epoch}/{epoch + 1}"] + [
        round(num, 4) for num in num_metrics
    ]

    return best_model, return_metrics


def save_results(
    train_cfg, data_cfg, model: nn.Module, metrics: list, exp_name: str, save_dir: str
):
    summary = {
        "train_config": train_cfg,
        "data_config": data_cfg,
        "student_metrics": {
            "epochs": metrics[0],
            "best_vl_total_loss": metrics[1],
            "best_vl_pocket_loss": metrics[2],
            "best_vl_aff_loss": metrics[3],
            "best_vl_DCC": metrics[4],
            "best_vl_DCC_SR": metrics[5],
            "best_vl_DVO": metrics[6],
            "best_DCC_nan_count": metrics[7],
            "best_vl_PCC": metrics[8],
            "avg_nan_count": metrics[9],
        },
        "teacher_metrics": {
            "best_vl_PCC": metrics[10],
            "best_vl_DCC_SR": metrics[11],
            "best_vl_aff_loss": metrics[12],
        },
    }

    # Save as .json file
    with open(
        os.path.join(train_cfg["save_dir"], f"{exp_name}_results.json"), "w"
    ) as fp:
        json.dump(summary, fp, indent=4)

    print(f"\nTraining completed. Results:")
    print(f"  Best epoch / Total epochs: {metrics[0]}")
    print(f"  Best validation total loss: {metrics[1]:.4f}")
    print(f"  Best validation PCC: {metrics[8]:.4f}")
    print(f"  Best validation DCC SR: {metrics[5]:.4f}\n")

    print(
        f"Best model and result file have been saved to: {train_cfg['save_dir']}"
    )
    return exp_name


def main():
    data_cfg, train_cfg = get_configs()

    if train_cfg["seed"] != -1:
        fix_seed(train_cfg["seed"])

    mdl = MasterDataLoader(
        data_cfg, data_cfg["seed"], train_cfg["batch_size"], train_cfg["num_workers"]
    )
    tr_loader, vl_loader = mdl.get_tr_vl_loader()

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

    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    split_name = os.path.basename(train_cfg['data_config']).replace('data_config_', '').replace('.json', '')
    exp_name = f"distill-pocket-soft_{split_name}_s{train_cfg['seed']}_{timestamp}"

    best_model, metrics = train_model(
        model,
        tr_loader,
        vl_loader,
        train_cfg,
        voxel_size=data_cfg["voxel_size"],
        exp_name=exp_name,
    )

    model.load_state_dict(best_model)

    save_results(
        train_cfg,
        data_cfg,
        model,
        metrics,
        exp_name=exp_name,
        save_dir=train_cfg["save_dir"],
    )

    return exp_name


if __name__ == "__main__":
    main()
