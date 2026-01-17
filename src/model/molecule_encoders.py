import sys
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    radius_graph,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GCN, GAT, SchNet, AttentiveFP
from torch_scatter import scatter_mean

from .utils import PyGRMSNorm, CoorsNorm, fourier_encode_dist, create_mol_batch_indices

class IntegratedMolEncoder(nn.Module):
    def __init__(
        self,
        encoder_types: Literal["attnfp", "gcn", "gat", "schnet", "egnn"],
        in_channels: int = 54,
        hidden_channels: int = 96,
        out_channels: int = 64,
        num_layers: int = 2,
        readout: Literal["mean", "sum", "max", "add"] = "mean",
        activation: str = "mish",
        **kwargs,
    ):
        super().__init__()

        self.encoder_dicts = nn.ModuleDict()
        self.encoder_dicts["2D_encoders"] = nn.ModuleList()
        self.encoder_dicts["3D_encoders"] = nn.ModuleList()
        
        # List to store learnable query tokens for 3D encoders (same order as 3D_encoders)
        self.attn_queries = nn.ParameterList()

        if "attnfp" in encoder_types:
            self.attnfp = MoleculeEncoder_AttnFP(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                edge_dim=1,
                num_timesteps=2,
                activation=activation
            )
            self.encoder_dicts["2D_encoders"].append(self.attnfp)

        if "gcn" in encoder_types:
            self.gcn = MoleculeEncoder_GCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                readout=readout,
                activation=activation
            )
            self.encoder_dicts["2D_encoders"].append(self.gcn)

        if "gat" in encoder_types:
            self.gat = MoleculeEncoder_GAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                heads=4,
                readout=readout,
                activation=activation
            )
            self.encoder_dicts["2D_encoders"].append(self.gat)

        if "schnet" in encoder_types:
            self.schnet = MoleculeEncoder_CustomSchNet(
                hidden_channels=hidden_channels,
                output_dim=out_channels,
                num_filters=kwargs["num_filters_schnet"],
                num_gaussians=kwargs["edge_num_gaussian_schnet"],
                num_interactions=kwargs["num_interactions_3d"],
                cutoff=kwargs["cutoff_3d"],
                readout=readout,
                dropout_rate=kwargs["dropout_3d"],
                activation=activation
            )

            # Cross-attention components: learnable query token
            self.schnet_conf_attn = nn.Sequential(
                nn.RMSNorm(out_channels),
                nn.Linear(out_channels, out_channels * 2)  # Key + Value projection
            )

            # Add to 3D encoders list
            self.encoder_dicts["3D_encoders"].append(
                nn.ModuleList([self.schnet, self.schnet_conf_attn])
            )
            # Add corresponding query token (same order)
            self.attn_queries.append(nn.Parameter(torch.randn(1, 1, out_channels)))

        if "egnn" in encoder_types:
            self.egnn = MoleculeEncoder_CustomEGNN(
                hidden_channels=hidden_channels,
                output_dim=out_channels,
                num_interactions=kwargs["num_interactions_3d"],
                cutoff=kwargs["cutoff_3d"],
                update_coors=True,
                update_feats=True,
                fourier_features=kwargs["edge_num_fourier_feats_egnn"],
                norm_feats=False,
                norm_coors=False,
                dropout=kwargs["dropout_3d"],
                readout=readout,
                soft_edge=kwargs["soft_edge_egnn"],
                activation=activation
            )

            # Cross-attention components: learnable query token
            self.egnn_conf_attn = nn.Sequential(
                nn.RMSNorm(out_channels),
                nn.Linear(out_channels, out_channels * 2)  # Key + Value projection
            )

            # Add to 3D encoders list
            self.encoder_dicts["3D_encoders"].append(
                nn.ModuleList([self.egnn, self.egnn_conf_attn])
            )
            # Add corresponding query token (same order)
            self.attn_queries.append(nn.Parameter(torch.randn(1, 1, out_channels)))

    def forward(self, mol_data, return_conf_data=False):
        x, edge_index, edge_attr, batch_index, z, pos = (
            mol_data.x,
            mol_data.edge_index,
            mol_data.edge_attr.unsqueeze(1),
            mol_data.batch,
            mol_data.z.clone(),
            mol_data.pos,
        )
        
        encoder_outputs = []

        if return_conf_data:
            conf_data = {
                'attn_weights': {},
                'egnn_pos': {}
            }

        if self.encoder_dicts["2D_encoders"]:
            for enc_2d in self.encoder_dicts["2D_encoders"]:
                encoder_outputs.append(
                    enc_2d(
                        x, edge_index, edge_attr, batch_index
                    )
                )

        if self.encoder_dicts["3D_encoders"]:
            pos = pos.permute((1, 0, 2))
            num_confs = pos.shape[0]
            z_repeated = z.repeat(num_confs)
            batch_index_repeated = create_mol_batch_indices(
                batch_index, num_confs
            )
            batch_size = batch_index.max().item() + 1
            pos_stacked = pos.reshape(-1, 3)

            if return_conf_data:
                conf_data['egnn_pos']['initial'] = pos_stacked.detach().cpu()

            for idx, enc_attn_pair in enumerate(self.encoder_dicts["3D_encoders"]):
                enc_3d, attn_3d = enc_attn_pair

                if isinstance(enc_3d, MoleculeEncoder_CustomEGNN) and return_conf_data:
                    enc_3d_out, pos_final = enc_3d(z_repeated, pos_stacked, batch_index_repeated, return_final_pos=True)
                    conf_data['egnn_pos']['final'] = pos_final.detach().cpu()
                else:
                    enc_3d_out = enc_3d(z_repeated, pos_stacked, batch_index_repeated)

                enc_3d_per_conf = enc_3d_out.split(batch_size, dim=0)
                enc_3d_stack = torch.stack(enc_3d_per_conf, dim=1)  # [batch_size, num_confs, out_channels]

                # Get corresponding query token (same index as encoder)
                attn_query = self.attn_queries[idx]
                
                # Cross-attention: learnable query token attending to conformer features
                q = attn_query.expand(batch_size, -1, -1)  # [batch_size, 1, out_channels]
                kv = attn_3d(enc_3d_stack)  # [batch_size, num_confs, out_channels * 2]
                k, v = kv.chunk(2, dim=-1)  # Each: [batch_size, num_confs, out_channels]
                
                # Compute attention scores (following cross_attention.py style)
                scale = k.size(-1) ** -0.5
                logits = (q @ k.transpose(-2, -1)) * scale  # [batch_size, 1, num_confs]
                attn = F.softmax(logits, dim=-1)

                if return_conf_data:
                    encoder_name = enc_3d.__class__.__name__.replace('MoleculeEncoder_Custom', '').lower()
                    conf_data['attn_weights'][encoder_name] = attn.detach().cpu()
                
                # Compute attention output
                weighted_sum = (attn @ v).squeeze(1)  # [batch_size, out_channels]

                encoder_outputs.append(weighted_sum)

            mol_features = torch.stack(encoder_outputs, dim=1)

        if return_conf_data:
            return mol_features, conf_data
        else:
            return mol_features

class MoleculeEncoder_CustomEGNN(nn.Module):
    """
    Customized molecular structure encoder based on EGNN adapted from:
    https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch_geometric.py
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        output_dim: int = 64,
        num_interactions: int = 6,
        m_dim: int = 16,
        fourier_features: int = 0,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = "add",
        update_coors: bool = True,
        update_feats: bool = True,
        norm_feats: bool = False,
        norm_coors: bool = False,
        dropout: float = 0.0,
        soft_edge: bool = False,
        activation: str = "mish"
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.update_coors = update_coors

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0)

        self.radius_graph = lambda pos, batch: radius_graph(
            pos, r=cutoff, batch=batch, max_num_neighbors=max_num_neighbors
        )  # partial func

        # EGNN layers (repeats num_interactions)
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            layer = EGNN_Layer(
                feats_dim=hidden_channels,
                pos_dim=3,
                m_dim=m_dim,
                fourier_features=fourier_features,
                update_coors=update_coors,
                update_feats=update_feats,
                norm_feats=norm_feats,
                norm_coors=norm_coors,
                dropout=dropout,
                soft_edge=soft_edge,
                aggr="mean",
            )
            self.interactions.append(layer)
        self.rms_norm_layers = nn.ModuleList(
            [PyGRMSNorm(hidden_dim=hidden_channels) for _ in range(num_interactions)]
        )
        
        self.lin1 = nn.Linear(hidden_channels, (hidden_channels + output_dim) // 2)
        self.dropout_1 = nn.Dropout(dropout)
        self.rms_norm_2 = PyGRMSNorm(hidden_dim=(hidden_channels + output_dim) // 2)
        self.lin2 = nn.Linear((hidden_channels + output_dim) // 2, output_dim)

        if activation == "mish":
            self.act = nn.Mish()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['mish', 'silu', 'elu']")

        if readout == "add":
            self.readout = global_add_pool
        elif readout == "mean":
            self.readout = global_mean_pool
        elif readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError(f"Unsupported readout: '{readout}'. Supported: ['add', 'mean', 'max]")

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.apply(interaction.init_)
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch=None, return_final_pos=False):
        """
        Forward pass compatible with SchNet interface

        Args:
            z: Atomic numbers [num_atoms]
            pos: Atomic coordinates [num_atoms, 3]
            batch: Batch indices [num_atoms]

        Returns:
            Molecular embeddings [batch_size, output_dim]
        """
        batch = torch.zeros_like(z) if batch is None else batch

        # Initial embedding
        h = self.embedding(z)

        if not self.update_coors:
            edge_index = self.radius_graph(pos, batch)

        # EGNN interaction layers
        for i, interaction in enumerate(self.interactions):
            if self.update_coors:
                edge_index = self.radius_graph(pos, batch)

            # Apply EGNN layer
            h = self.rms_norm_layers[i](h, batch)
            h, pos = interaction(h, pos, edge_index, batch=batch)
            if h.isnan().any() or h.isinf().any():
                send_message_slack("EGNN h 에서 nan, inf 발생")
                # breakpoint()

        # Final processing
        h1 = self.lin1(h)
        h2 = self.act(h1)
        h3 = self.dropout_1(h2)
        h4 = self.lin2(h3)

        out = self.readout(h4.float(), batch)
        if out.isnan().any() or out.isinf().any():
            send_message_slack("EGNN out 에서 nan, inf 발생")
            # breakpoint()

        if return_final_pos:
            return out, pos
        else:
            return out


class MoleculeEncoder_CustomSchNet(SchNet):
    """
    The continuous-filter convolutional neural network SchNet from the
    “SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions” paper
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        output_dim: int = 64,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph=None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dropout_rate: float = 0.1,
        activation="mish"
    ):
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
        )

        self.embedding = nn.Embedding(100, hidden_channels, padding_idx=0, max_norm=1.0)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.interaction_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_interactions)]
        )
        self.rms_norm_layers = nn.ModuleList(
            [PyGRMSNorm(hidden_dim=hidden_channels) for _ in range(num_interactions)]
        )

        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.interaction_dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_interactions)]
        )

        if activation == "mish":
            self.act = nn.Mish()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['mish', 'silu', 'elu']")

        self.lin1 = nn.Linear(hidden_channels, (hidden_channels + output_dim) // 2)
        self.rms_norm_2 = PyGRMSNorm(hidden_dim=(hidden_channels + output_dim) // 2)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.lin2 = nn.Linear((hidden_channels + output_dim) // 2, output_dim)
        self.reset_parameters()

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch=None) -> torch.Tensor:
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        h = self.embedding_dropout(h)

        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)
        
        for i, interaction in enumerate(self.interactions):
            h = self.rms_norm_layers[i](h, batch)
            h_interaction = interaction(h, edge_index, edge_weight, edge_attr)
            h_interaction = self.interaction_dropouts[i](
                h_interaction
            )
            h = h + h_interaction

        h1 = self.lin1(h)
        h2 = self.rms_norm_2(h1, batch)
        h3 = self.act(h2)
        h4 = self.dropout_1(h3)
        h5 = self.lin2(h4)
        out = self.readout(h5, batch, dim=0)
        if out.isnan().any() or out.isinf().any():
                send_message_slack("schnet h 에서 nan, inf 발생")
                # breakpoint()
        return out


class MoleculeEncoder_GCN(nn.Module):
    def __init__(
        self, in_channels=54, hidden_channels=32, out_channels=64, num_layers=2, readout="mean", activation="mish"
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
        )
        
        if activation == "mish":
            self.act = nn.Mish()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['mish', 'silu', 'elu']")

        if readout == "add":
            self.readout = global_add_pool
        elif readout == "mean":
            self.readout = global_mean_pool
        elif readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError(f"Unsupported readout: {readout}. Supported: ['add', 'mean', 'max']")


    def forward(self, x, edge_index, edge_attr=None, batch_index=None):
        # edge_attr is dummy arguments (not using)
        out = self.gcn(x, edge_index)
        out = self.act(out)
        out = self.readout(out, batch_index)

        return out


class MoleculeEncoder_AttnFP(nn.Module):
    def __init__(
        self,
        in_channels=54,
        hidden_channels=32,
        out_channels=64,
        num_layers=2,
        edge_dim=1,
        num_timesteps=2,
        activation="mish"
    ):
        super().__init__()

        self.attenfp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            edge_dim=edge_dim,
            num_timesteps=num_timesteps,
        )

        if activation == "mish":
            self.act = nn.Mish()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['mish', 'silu', 'elu']")

    def forward(self, x, edge_index, edge_attr, batch_index):

        out = self.attenfp(x, edge_index, edge_attr, batch_index)
        out = self.act(out)

        return out


class MoleculeEncoder_GAT(nn.Module):
    def __init__(
        self, in_channels=54, hidden_channels=32, out_channels=64, num_layers=2, heads=4, readout="mean", activation="mish"
    ):
        super().__init__()

        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            v2=True,
            heads=heads,
        )

        if activation == "mish":
            self.act = nn.Mish()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            raise ValueError(f"Invalid activation function '{activation}'. Supported: ['mish', 'silu', 'elu']")

        if readout == "add":
            self.readout = global_add_pool
        elif readout == "mean":
            self.readout = global_mean_pool
        elif readout == "max":
            self.readout = global_max_pool
        else:
            raise ValueError(f"Unsupported readout: {readout}. Supported: ['add', 'mean', 'max]")


    def forward(self, x, edge_index, edge_attr=None, batch_index=None):
        out = self.gat(x, edge_index, edge_attr)
        out = self.act(out)
        out = self.readout(out, batch_index)

        return out


class EGNN_Layer(MessagePassing):
    """Single EGNN layer"""

    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=16,
        fourier_features=0,
        soft_edge=False,
        norm_feats=False,
        norm_coors=False,
        norm_coors_scale_init=1e-2,
        update_feats=True,
        update_coors=True,
        dropout=0.0,
        coor_weights_clamp_value=None,
        aggr="add",
        **kwargs,
    ):
        assert aggr in {
            "add",
            "sum",
            "max",
            "mean",
        }, "pool method must be a valid option"
        assert (
            update_feats or update_coors
        ), "you must update either features, coordinates, or both"
        kwargs.setdefault("aggr", aggr)
        super(EGNN_Layer, self).__init__(**kwargs)

        # model params
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = coor_weights_clamp_value

        self.edge_input_dim = (
            (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # EDGES
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.Mish(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.Mish(),
        )

        self.edge_weight = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None
        )

        # NODES
        self.node_norm = PyGRMSNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init)

        self.node_mlp = (
            nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 2),
                self.dropout,
                nn.Mish(),
                nn.Linear(feats_dim * 2, feats_dim),
            )
            if update_feats
            else None
        )

        # COORDS
        self.coors_mlp = (
            nn.Sequential(
                nn.Linear(m_dim, m_dim * 4),
                self.dropout,
                nn.Mish(),
                nn.Linear(m_dim * 4, 1),
            )
            if update_coors
            else None
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std=1e-3)
            nn.init.zeros_(module.bias)

    def forward(self, h, coors, edge_index, edge_attr=None, batch=None):
        """Forward pass with separate h (features) and coors (coordinates)"""
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            rel_dist_encoded = fourier_encode_dist(
                rel_dist, num_encodings=self.fourier_features
            )
            rel_dist_encoded = rel_dist_encoded.squeeze(1)  # Remove middle dim
        else:
            rel_dist_encoded = rel_dist

        # Get edge attributes for message passing
        if edge_attr is not None:
            edge_attr_feats = torch.cat([edge_attr, rel_dist_encoded], dim=-1)
        else:
            edge_attr_feats = rel_dist_encoded

        # Message passing
        hidden_out, coors_out = self.propagate(
            edge_index,
            x=h,
            edge_attr=edge_attr_feats,
            coors=coors,
            rel_coors=rel_coors,
            batch=batch,
        )

        return hidden_out, coors_out

    def message(self, x_i, x_j, edge_attr):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index, size=None, **kwargs):
        """Custom propagate to handle both features and coordinates"""
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        update_kwargs = self.inspector.collect_param_data("update", coll_dict)

        # Get messages
        m_ij = self.message(**msg_kwargs)

        # Update coordinates if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)

            # Clamp if specified
            if self.coor_weights_clamp_value:
                coor_wij.clamp_(
                    min=-self.coor_weights_clamp_value,
                    max=self.coor_weights_clamp_value,
                )

            # Normalize if needed
            if self.norm_coors:
                kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])

            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        # Update features if specified
        if self.update_feats:
            # Weight the edges if soft edge is enabled
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)

            m_i = self.aggregate(m_ij, **aggr_kwargs)

            hidden_feats = (
                self.node_norm(kwargs["x"], kwargs["batch"])
                if self.node_norm
                else kwargs["x"]
            )
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]

        return self.update((hidden_out, coors_out), **update_kwargs)