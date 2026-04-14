"""
Original code from MONAI (https://github.com/Project-MONAI/MONAI)
Licensed under the Apache License, Version 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

########## in-house codes ###########
from .utils import RMSNorm3d


class UnetrBasicBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        drop_rate=0.0,
        activation="gelu"
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.norm1 = RMSNorm3d(out_channels)
        self.norm2 = RMSNorm3d(out_channels)
        self.dropout = nn.Dropout(drop_rate)

        if activation=="gelu":
           self.act = nn.GELU()
        elif activation=="silu":
           self.act = nn.SiLU()
        elif activation=="relu":
           self.act = nn.ReLU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['gelu', 'silu', 'relu']")

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        skip_channels,
        kernel_size=3,
        upsample_kernel_size=2,
        res_block=False,
        drop_rate=0.0,
        activation="gelu"
    ):
        super().__init__()
        self.transp_conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
        )

        self.skip_transform = nn.Conv3d(skip_channels, out_channels, kernel_size=1)

        if res_block:
            self.conv_block = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels * 2,  # upsampling + skip connection
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate,
                activation=activation
            )
        else:
            self.conv_block = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                drop_rate=drop_rate,
                activation=activation
            )

    def forward(self, x, skip=None):
        x_up = self.transp_conv(x)

        if skip is not None:
            if skip.shape[1] != x_up.shape[1]:
                skip = self.skip_transform(skip)

            if x_up.shape[2:] != skip.shape[2:]:
                d, h, w = x_up.shape[2], x_up.shape[3], x_up.shape[4]
                skip = F.interpolate(skip, size=(d, h, w), mode="nearest")

            x_up = torch.cat((x_up, skip), dim=1)

        result = self.conv_block(x_up)
        return result


class UnetOutBlock(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, drop_rate, activation="gelu"):
        super().__init__()
        expansion_ratio = 4/3
        hidden_size = int(in_channels * expansion_ratio)
        
        self.conv1 = nn.Conv3d(in_channels, hidden_size, kernel_size=3, padding=1)
        self.norm1 = RMSNorm3d(hidden_size)

        self.conv2 = nn.Conv3d(hidden_size, hidden_size//2, kernel_size=3, padding=1)
        self.norm2 = RMSNorm3d(hidden_size//2)

        self.conv3 = nn.Conv3d(hidden_size//2, out_channels, kernel_size=1)

        self.residual_proj = nn.Conv3d(in_channels, hidden_size//2, kernel_size=1)
        self.dropout = nn.Dropout(drop_rate)

        if activation=="gelu":
           self.act = nn.GELU()
        elif activation=="silu":
           self.act = nn.SiLU()
        elif activation=="relu":
           self.act = nn.ReLU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['gelu', 'silu', 'relu']")

    def forward(self, x):
        residual = x

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)

        h = h + self.residual_proj(residual)
        h = self.act(h)
        h = self.dropout(h)
        
        out = self.conv3(h)
        return out
