"""
Original code (SwinUNETR) from MONAI (https://github.com/Project-MONAI/MONAI)
Licensed under the Apache License, Version 2.0
Modified for developing binding affinity predicting model
"""

import sys
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

########## in-house codes ###########
from .encoder import SwinTransformer
from .decoder import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from .molecule_encoders import IntegratedMolEncoder
from .cross_attention import (
    CrossAttention_P2L,
    CrossAttention_L2P,
    SpatialAttentionPooling,
    SequentialAttentionPooling,
    RegressionHead,
)
from .utils import ensure_tuple_rep


class InSiteDTA(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 21,
        out_channels: int = 22,
        feature_size: int = 48,
        # SwinTransformer params
        sw_depths: Sequence[int] = (2, 2, 2),
        sw_window_size: Sequence[int] = (7, 7, 7),
        sw_patch_size: Sequence[int] = (2, 2, 2),
        sw_num_heads: Sequence[int] = (3, 6, 12),
        sw_mlp_ratio: float = 4.0,
        sw_qkv_bias: bool = True,
        sw_drop_rate: float = 0.0,
        sw_attn_drop_rate: float = 0.0,
        sw_norm_layer=nn.RMSNorm,
        sw_drop_path_rate: float = 0.0,
        sw_act: Literal["gelu", "relu", "swiglu"] = "gelu",
        # Molecule encoders params
        mol_encoder_types=["attnfp", "schnet", "egnn"],
        mol_in_channels=54,
        mol_hidden_channels=128,
        mol_out_channels=128,  # 4³
        mol_num_layers=2,
        mol_num_interactions_3d=3,
        mol_dropout_3d=0.15,
        mol_readout="mean",
        mol_cutoff_3d=10,
        mol_num_filters_schnet=128,
        mol_edge_num_gaussian_schnet=50,
        mol_edge_num_fourier_feats_egnn=3,
        mol_soft_edge_egnn=False,
        mol_act: Literal["mish", "silu", "elu"] = "mish",
        # Decoder params
        dec_drop_rate=0.0,
        dec_act: Literal["gelu", "silu", "relu"] = "gelu",
        # Cross Attn params
        ca_num_heads=8,
        ca_dropout=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sw_window_size = ensure_tuple_rep(sw_window_size, spatial_dims)
        self.hidden_sizes = [feature_size * (2**i) for i in range(len(sw_depths))]
        self.mol_feature_dim = mol_out_channels

        self.swinViT = SwinTransformer(
            in_channels=in_channels,
            spatial_dims=spatial_dims,
            hidden_sizes=self.hidden_sizes,
            depths=sw_depths,
            window_size=self.sw_window_size,
            patch_size=sw_patch_size,
            num_heads=sw_num_heads,
            mlp_ratio=sw_mlp_ratio,
            qkv_bias=sw_qkv_bias,
            drop_rate=sw_drop_rate,
            attn_drop_rate=sw_attn_drop_rate,
            norm_layer=sw_norm_layer,
            drop_path_rate=sw_drop_path_rate,
            activation=sw_act
        )

        self.mol_encoder = IntegratedMolEncoder(
            encoder_types=mol_encoder_types,
            in_channels=mol_in_channels,
            hidden_channels=mol_hidden_channels,
            out_channels=mol_out_channels,
            num_layers=mol_num_layers,
            dropout_3d=mol_dropout_3d,
            num_interactions_3d=mol_num_interactions_3d,
            cutoff_3d=mol_cutoff_3d,
            num_filters_schnet=mol_num_filters_schnet,
            edge_num_gaussian_schnet=mol_edge_num_gaussian_schnet,
            edge_num_fourier_feats_egnn=mol_edge_num_fourier_feats_egnn,
            soft_edge_egnn=mol_soft_edge_egnn,
            activation=mol_act
        )


        self.cross_attention_bottleneck_p2l = CrossAttention_P2L(
            patch_dim=self.hidden_sizes[2],
            molecule_dim=self.mol_feature_dim,
            hidden_dim=self.hidden_sizes[2],
            num_heads=ca_num_heads,
            dropout_rate=ca_dropout
        )

        self.cross_attention_bottleneck_l2p = CrossAttention_L2P(
            patch_dim=self.hidden_sizes[2],
            molecule_dim=self.mol_feature_dim,
            hidden_dim=self.hidden_sizes[2],
            num_heads=ca_num_heads,
        )

        self.pocket_encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=self.hidden_sizes[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GELU(),
        )

        self.cross_attention_l2pocket = CrossAttention_L2P(
            patch_dim=self.hidden_sizes[0], molecule_dim=self.mol_feature_dim, hidden_dim=128, num_heads=ca_num_heads
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[0],
            out_channels=self.hidden_sizes[0],
            kernel_size=3,
            stride=1,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[1],
            out_channels=self.hidden_sizes[1],
            kernel_size=3,
            stride=1,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[2],
            out_channels=self.hidden_sizes[2],
            kernel_size=3,
            stride=1,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[2],
            out_channels=self.hidden_sizes[1],
            skip_channels=self.hidden_sizes[2],
            kernel_size=3,
            upsample_kernel_size=2,
            res_block=True,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[1],
            out_channels=self.hidden_sizes[0],
            skip_channels=self.hidden_sizes[1],
            kernel_size=3,
            upsample_kernel_size=2,
            res_block=True,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[0],
            out_channels=self.hidden_sizes[0],
            skip_channels=self.hidden_sizes[0],
            kernel_size=3,
            upsample_kernel_size=2,
            res_block=True,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_sizes[0],
            out_channels=out_channels,
            drop_rate=dec_drop_rate,
            activation=dec_act
        )

        self.attention_pooling = SpatialAttentionPooling(
            dim=self.hidden_sizes[2]
        )

        self.seq_attn_pooling = SequentialAttentionPooling(dim=self.mol_feature_dim)

        self.regression_head = RegressionHead(
            in_dim=self.hidden_sizes[2] + self.mol_feature_dim,
            hidden_dim=1024,
            out_dim=1,
        )

    def forward(self, x_in, mol_data=None, return_conf_data=False, return_attn_map=False):

        mol_features = None
        if mol_data is not None:
            if return_conf_data:
                mol_features, conf_data = self.mol_encoder(mol_data, return_conf_data=True)
            else:
                mol_features = self.mol_encoder(mol_data)

        hidden_states = self.swinViT(x_in)
        bottleneck = hidden_states[-1]

        if mol_features is not None:
            bottleneck_p2l = self.cross_attention_bottleneck_p2l(
                patch_features=bottleneck, mol_features=mol_features
            )

        enc1 = self.encoder1(hidden_states[0])
        enc2 = self.encoder2(hidden_states[1])
        enc3 = self.encoder3(hidden_states[2])
        # enc4 = self.encoder3(hidden_states[3])

        dec3 = self.decoder3(bottleneck_p2l, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        logits = self.out(dec1)
        pred_pocket_mask = F.sigmoid(logits[:, -1:, ...])
        
        dec1_pred_pocket = dec1 * pred_pocket_mask.detach()
        # x_in_enc_pred_pocket = self.pocket_encoder(x_in_pred_pocket)

        if return_attn_map:
            l2pocket, l2pocket_attn = self.cross_attention_l2pocket(
                patch_features=dec1_pred_pocket, 
                mol_features=mol_features,
                return_attn_map=True
            )
        else:
            l2pocket = self.cross_attention_l2pocket(
                patch_features=dec1_pred_pocket, mol_features=mol_features
            )

        # 병렬 경로
        pooled_features_p2l = self.attention_pooling(bottleneck_p2l)
        pooled_features_l2pocket = self.seq_attn_pooling(l2pocket)


        pooled_features = torch.cat(
            [pooled_features_p2l, pooled_features_l2pocket], dim=1
        )
        regression_output = self.regression_head(pooled_features).squeeze()
        
        if return_conf_data and return_attn_map:
            return logits, regression_output, conf_data, l2pocket_attn
        elif return_conf_data:
            return logits, regression_output, conf_data
        elif return_attn_map:
            return logits, regression_output, l2pocket_attn
        else:
            return logits, regression_output