from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
    
    def forward(self, x):
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class PyGRMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x, batch):
        x_squared = x.pow(2)
        rms_squared = scatter_mean(x_squared, batch, dim=0)[batch]
        rms = torch.sqrt(rms_squared.mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RMSNorm3d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=(2, 3, 4), keepdim=True) + self.eps)
        x_normed = x / rms
        weight = self.weight.view(1, -1, 1, 1, 1)

        return x_normed * weight


class ShiftedSoftplus(nn.Module):
    """Schnet vanilla act. function"""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) - self.shift


def ensure_tuple_rep(val, dim):
    if isinstance(val, Sequence):
        return tuple(val)
    return tuple(val for _ in range(dim))


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7)
            .contiguous()
            .view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    else:  # len(x_shape) == 4
        b, h, w, c = x.shape
        x = x.view(
            b,
            h // window_size[0],
            window_size[0],
            w // window_size[1],
            window_size[1],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size[0] * window_size[1], c)
        )

    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(
            b,
            h // window_size[0],
            w // window_size[1],
            window_size[0],
            window_size[1],
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """
    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for h in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                for w in (
                    slice(-window_size[2]),
                    slice(-window_size[2], -shift_size[2]),
                    slice(-shift_size[2], None),
                ):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in (
            slice(-window_size[0]),
            slice(-window_size[0], -shift_size[0]),
            slice(-shift_size[0], None),
        ):
            for w in (
                slice(-window_size[1]),
                slice(-window_size[1], -shift_size[1]),
                slice(-shift_size[1], None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def embedd_token(x, embedding_dims, emb_layers):
    """Apply embeddings to discrete tokens"""
    if len(embedding_dims) == 0:
        return x

    # Assume first few columns are discrete tokens to embed
    embedded_parts = []
    continuous_start_idx = 0

    for i, (emb_layer) in enumerate(emb_layers):
        # Extract discrete token column
        token_col = x[:, continuous_start_idx].long()
        embedded = emb_layer(token_col)
        embedded_parts.append(embedded)
        continuous_start_idx += 1

    # Concatenate embedded tokens with continuous features
    if continuous_start_idx < x.shape[1]:
        continuous_features = x[:, continuous_start_idx:]
        if embedded_parts:
            return torch.cat(embedded_parts + [continuous_features], dim=-1)
        else:
            return continuous_features
    else:
        return torch.cat(embedded_parts, dim=-1)


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """Fourier encode distances"""
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1) * scale_init)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


def create_mol_batch_indices(batch_index, num_confs):
    batch_size = batch_index.max().item() + 1

    # Vectorized approach
    conf_offsets = torch.arange(num_confs, device=batch_index.device) * batch_size
    batch_index_expanded = batch_index.unsqueeze(0).expand(
        num_confs, -1
    )  # [num_confs, N]
    conf_offsets_expanded = conf_offsets.unsqueeze(1).expand(
        -1, len(batch_index)
    )  # [num_confs, N]

    batch_index_repeated = (batch_index_expanded + conf_offsets_expanded).flatten()
    return batch_index_repeated


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [B, 1, 1, 1, 1]
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
