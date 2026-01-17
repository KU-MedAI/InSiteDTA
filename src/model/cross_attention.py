import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import RMSNorm3d


class CrossAttention_P2L(nn.Module):
    """Cross attention between patch features (query) and molecular features (key/value)"""

    def __init__(
        self, patch_dim, molecule_dim, hidden_dim, num_heads=8, dropout_rate=0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Layers for 3D positional encoding
        self.pos_encoding_3d = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.patch_to_q = nn.Linear(patch_dim, hidden_dim)
        self.molecule_to_kv = nn.Linear(molecule_dim, hidden_dim * 2)

        self.proj = nn.Linear(hidden_dim, patch_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        self.norm_molecules = nn.RMSNorm(molecule_dim)
        self.norm_patches = nn.RMSNorm(patch_dim)

        self.attn_drop = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(patch_dim, patch_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(patch_dim * 4, patch_dim),
        )

        self.ffn_norm = nn.RMSNorm(patch_dim)

    def forward(self, patch_features, mol_features):
        """
        Args:
            patch_features: Patch features [B, C, D, H, W] - 3D tensor
            mol_features: Molecular features [B, n_encoders, M] - outputs from n_encoders encoders
        Returns:
            Updated patch features [B, C, D, H, W]
        """
        # Transform patch features shape and normalization
        if len(patch_features.shape) == 5:  # 3D data
            B, C, D, H, W = patch_features.shape
            # Generate 3D coordinate grid
            coords_d = (
                torch.arange(D, device=patch_features.device, dtype=torch.float32) / D
            )
            coords_h = (
                torch.arange(H, device=patch_features.device, dtype=torch.float32) / H
            )
            coords_w = (
                torch.arange(W, device=patch_features.device, dtype=torch.float32) / W
            )

            grid_d, grid_h, grid_w = torch.meshgrid(
                coords_d, coords_h, coords_w, indexing="ij"
            )
            pos_coords = torch.stack([grid_d, grid_h, grid_w], dim=-1)
            pos_coords = pos_coords.reshape(-1, 3).unsqueeze(0).repeat(B, 1, 1)

            # Compute positional encoding
            pos_encoding = self.pos_encoding_3d(pos_coords)

            patch_features_reshape = patch_features.permute(0, 2, 3, 4, 1)
            patch_features_flat = patch_features_reshape.reshape(B, D * H * W, C)
        elif len(patch_features.shape) == 4:  # 2D data support planned
            # B, C, H, W = patch_features.shape
            # patch_features_reshape = patch_features.permute(0, 2, 3, 1)     # [B, H, W, C]
            # patch_features_flat = patch_features_reshape.reshape(B, H*W, C)      # [B, H*W, C]
            pass

        # Normalization
        patch_features_flat = self.norm_patches(patch_features_flat)
        mol_features = self.norm_molecules(mol_features)

        # Transform to Query, Key, Value
        q = self.patch_to_q(patch_features_flat)
        q = q + pos_encoding
        kv = self.molecule_to_kv(mol_features)
        k, v = kv.chunk(2, dim=-1)

        # Multi-head transformation
        N = patch_features_flat.size(1)
        n_encoders = mol_features.size(1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, n_encoders, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, n_encoders, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_dim)

        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual
        attended = out + patch_features_flat

        # Feed forward network
        out = self.ffn_norm(attended)
        out = self.ffn(out)
        out = out + attended

        # Reconstruct to original shape
        if len(patch_features.shape) == 5:  # 3D data
            out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        elif len(patch_features.shape) == 4:  # 2D data
            out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return out


class CrossAttention_L2P(nn.Module):
    """Cross attention between molecular features (query) and patch features (key/value)"""

    def __init__(
        self, patch_dim, molecule_dim, hidden_dim, num_heads=8, dropout_rate=0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Layers for 3D positional encoding
        self.pos_encoding_3d = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self.molecule_to_q = nn.Linear(molecule_dim, hidden_dim)
        self.patch_to_kv = nn.Linear(patch_dim, hidden_dim * 2)

        self.proj = nn.Linear(hidden_dim, molecule_dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        self.norm_molecules = nn.RMSNorm(molecule_dim)
        self.norm_patches = nn.RMSNorm(patch_dim)

        self.attn_drop = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(molecule_dim, molecule_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(molecule_dim * 4, molecule_dim),
        )

        self.ffn_norm = nn.RMSNorm(molecule_dim)

    def forward(self, patch_features, mol_features, return_attn_map=False):
        """
        Args:
            patch_features: Patch features [B, C, D, H, W] - 3D tensor
            mol_features: Molecular features [B, n_encoders, M] - outputs from n_encoders encoders
        Returns:
            Updated molecular features [B, n_encoders, M]
        """
        # Transform patch features shape and normalization
        if len(patch_features.shape) == 5:  # 3D data
            B, C, D, H, W = patch_features.shape
            # Generate 3D coordinate grid
            coords_d = (
                torch.arange(D, device=patch_features.device, dtype=torch.float32) / D
            )
            coords_h = (
                torch.arange(H, device=patch_features.device, dtype=torch.float32) / H
            )
            coords_w = (
                torch.arange(W, device=patch_features.device, dtype=torch.float32) / W
            )

            grid_d, grid_h, grid_w = torch.meshgrid(
                coords_d, coords_h, coords_w, indexing="ij"
            )
            pos_coords = torch.stack([grid_d, grid_h, grid_w], dim=-1)
            pos_coords = pos_coords.reshape(-1, 3).unsqueeze(0).repeat(B, 1, 1)

            # Compute positional encoding
            pos_encoding = self.pos_encoding_3d(pos_coords)

            patch_features_reshape = patch_features.permute(0, 2, 3, 4, 1)
            patch_features_flat = patch_features_reshape.reshape(B, D * H * W, C)
        elif len(patch_features.shape) == 4:  # 2D data
            # B, C, H, W = patch_features.shape
            # patch_features_reshape = patch_features.permute(0, 2, 3, 1)     # [B, H, W, C]
            # patch_features_flat = patch_features_reshape.reshape(B, H*W, C)      # [B, H*W, C]
            pass

        # Normalization
        patch_features_flat = self.norm_patches(patch_features_flat)
        mol_features = self.norm_molecules(mol_features)

        # Transform to Query, Key, Value
        q = self.molecule_to_q(mol_features)
        kv = self.patch_to_kv(patch_features_flat)
        k, v = kv.chunk(2, dim=-1)

        # Add positional encoding
        k = k + pos_encoding
        v = v + pos_encoding

        # Multi-head transformation
        N = patch_features_flat.size(1)
        n_encoders = mol_features.size(1)

        q = q.reshape(B, n_encoders, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output
        out = (attn @ v).transpose(1, 2).reshape(B, n_encoders, self.hidden_dim)

        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual connection
        attended = out + mol_features

        # Feed forward network
        out = self.ffn_norm(attended)
        out = self.ffn(out)
        out = out + attended
        
        if return_attn_map:
            return out, attn  # ← attention 반환 추가
        return out


class SpatialAttentionPooling(nn.Module):
    """Attention-based pooling module that compresses 3D data to a 1D vector"""

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=1),
            RMSNorm3d(dim // 2),
            nn.GELU(),
            nn.Conv3d(dim // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        attention_weights = torch.sigmoid(self.attention(x))

        # Apply weights and global pooling
        x = x * attention_weights
        return torch.sum(x, dim=(2, 3, 4))


class SequentialAttentionPooling(nn.Module):
    """Attention-based pooling module that compresses sequential data to a 1D vector"""
    
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x).squeeze(-1), dim=1)
        return torch.sum(x * attention_weights.unsqueeze(-1), dim=1)

class RegressionHead(nn.Module):
    """MLP head for regression tasks"""

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            # nn.Linear(hidden_dim, hidden_dim // 4),
            # nn.LayerNorm(hidden_dim // 4),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)
