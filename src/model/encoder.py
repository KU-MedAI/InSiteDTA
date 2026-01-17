"""
Swin-Transformer Code based on: "Liu et al.,
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
<https://arxiv.org/abs/2103.14030>"
https://github.com/microsoft/Swin-Transformer
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from .utils import (
    DropPath,
    get_window_size,
    window_partition,
    window_reverse,
    compute_mask,
    SwiGLU
)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_dims: int,
        hidden_sizes: Sequence[int],
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int] = (2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.RMSNorm,
        drop_path_rate=0.0,
        activation="gelu"
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.hidden_sizes = hidden_sizes
        self.window_size = window_size
        self.patch_size = patch_size
        self.depths = depths
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_sizes[0],
            patch_norm=norm_layer,
            spatial_dims=spatial_dims,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Calculate drop path rates for all blocks; if depths=[2,2,2], there are 6 blocks total
        # Array with total_blocks values from 0 to drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        dpr_index = 0

        for i_layer in range(self.num_layers):
            curr_depth = depths[i_layer]

            layer_dpr = dpr[dpr_index : dpr_index + curr_depth]
            dpr_index += curr_depth

            layer = BasicLayer(
                depth=curr_depth,
                dim=hidden_sizes[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=layer_dpr,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                activation=activation
            )

            self.layers.append(layer)

    def forward(self, x, mol_features_list=None):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        features = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            features.append(x)

        return features


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Sequence[int] = (2, 2, 2),
        in_channels: int = 21,
        embed_dim: int = 48,
        spatial_dims: int = 3,
        patch_norm=nn.RMSNorm,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if len(patch_size) == 3:
            self.proj = nn.Conv3d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        elif len(patch_size) == 2:
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

        self.norm = patch_norm(embed_dim)

    def forward(self, x):
        x_shape = x.size()
        x = self.proj(x)
        if len(x_shape) == 5:
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        elif len(x_shape) == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.RMSNorm, spatial_dims=3):
        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 1::2, 0::2, :]
            x5 = x[:, 1::2, 0::2, 1::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat(
                [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))],
                -1,
            )

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        drop_path=[],
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.RMSNorm,
        downsample=None,
        activation="gelu"
    ):
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth

        if isinstance(drop_path, float):
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    activation=activation
                )
                for i in range(depth)
            ]
        )

        self.downsample = None
        if downsample is not None:
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size)
            )

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )
            x = x.permute(
                0, 2, 3, 4, 1
            ).contiguous()
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)

            for block in self.blocks:
                x = block(x, attn_mask)

            x = x.view(b, d, h, w, -1)

            if self.downsample is not None:
                x = self.downsample(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            x = x.permute(0, 2, 3, 1).contiguous()
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)

            for block in self.blocks:
                x = block(x, attn_mask)

            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = x.permute(0, 3, 1, 2).contiguous()

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=0.0,
        norm_layer=nn.RMSNorm,
        activation="gelu"
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_drop=drop_rate,
            attn_drop=attn_drop_rate,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, dropout_rate=drop_rate, activation=activation)

    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)

        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size(
                (d, h, w), self.window_size, self.shift_size
            )

            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]
        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size(
                (h, w), self.window_size, self.shift_size
            )
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        # window shift
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(
                    x,
                    shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                    dims=(1, 2, 3),
                )
            elif len(x_shape) == 4:
                shifted_x = torch.roll(
                    x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2)
                )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)

        # Restore shifted window
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(
                    shifted_x,
                    shifts=(shift_size[0], shift_size[1], shift_size[2]),
                    dims=(1, 2, 3),
                )
            elif len(x_shape) == 4:
                x = torch.roll(
                    shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2)
                )
        else:
            x = shifted_x

        # remove paddings
        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, proj_drop=0.0, attn_drop=0.0
    ):
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position encoding
        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1)
                    * (2 * self.window_size[1] - 1)
                    * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            coords = torch.stack(
                torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
            )
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (
                2 * self.window_size[2] - 1
            )
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads
                )
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask):
        # x -> (B * n_windows, window_size ** 3, channels)
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if (
            mask is not None
        ):  # if mask is None, it means calculating unshifted window attention
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.0, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.drop = nn.Dropout(dropout_rate)

        if activation=="gelu":
            self.act = nn.GELU()
        elif activation=="relu":
            self.act = nn.ReLU()
        elif activation=="swiglu":
            self.act = SwiGLU(mlp_dim)
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['gelu', 'relu', 'swiglu']")

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x