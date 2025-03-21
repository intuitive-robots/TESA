import einops
import torch
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)

from src.models.layer_utils import layer_factory


class GraphEmbedder(nn.Module):
    r"""
    Relations in Embeddings.
    Use Rie-PSG data.
    kopiert von: Paul Mattes
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layer,
        layer_type,
        pool_type,
        heads,
        dropout,
        edge_dim=None,
        do_res_norm=False,
    ):
        r"""
        @param layer_type: GCN|GAT|GATv2|FiLM|pool (use only pooling)|TransConv|pool_noFC (no fully connected layer)
        -> uses edge attributes for those layers for GAT,TransConv
        @param pool_type: mean|add|max

        """
        super(GraphEmbedder, self).__init__()
        self.edge_dim = edge_dim

        self.only_use_pooling = False
        self.use_edge_attr = layer_type in ["GAT", "GATv2", "TransConv"]
        self.do_res_norm = do_res_norm
        if pool_type == "mean":
            pool_class = global_mean_pool
        elif pool_type == "add":
            pool_class = global_add_pool
        elif pool_type == "max":
            pool_class = global_max_pool
        else:
            raise ValueError(f"pool_type {pool_type} not supported")
        self.pooling_layer = pool_class

        # setup layers
        if layer_type == "pool":
            self.only_use_pooling = True
        elif layer_type == "pool_noFC":
            self.only_use_pooling = True
            self.no_empty_params = nn.Linear(1, 1)
        else:
            self.input_layer = layer_factory(
                layer_type, input_dim, hidden_dim, heads, dropout, edge_dim
            )
            self.layers = ModuleList()
            for _ in range(num_layer):
                layer = layer_factory(
                    layer_type, hidden_dim, hidden_dim, heads, dropout, edge_dim
                )
                self.layers.append(layer)
            if self.do_res_norm:
                self.norms = ModuleList()
                for _ in range(num_layer):
                    self.norms.append(nn.LayerNorm(hidden_dim))
        if not layer_type == "pool_noFC":
            self.output = nn.Linear(hidden_dim, output_dim)
        else:
            self.output = nn.Identity()
        if self.only_use_pooling:
            if layer_type == "pool":
                self.projection_layer = nn.Linear(input_dim, hidden_dim)
            elif layer_type == "pool_noFC":
                self.projection_layer = nn.Identity()
            else:
                raise ValueError(f"layer_type {layer_type} not supported")

        self.leaky_relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr = (
            data.x.to(torch.float),
            data.edge_index,
            data.edge_attr.to(torch.float),
        )
        edge_attr = einops.rearrange(
            edge_attr, "(n dim) -> n dim", dim=self.edge_dim)
        edge_index = edge_index.to(torch.int64)

        if not self.only_use_pooling:
            if self.use_edge_attr:
                x = self.input_layer(x, edge_index, edge_attr)
            else:
                x = self.input_layer(
                    x, edge_index
                )  # posible alternative: use res_norm here as well?
            x = self.leaky_relu(x)
            if self.training:
                x = self.dropout(x)

            for i, layer in enumerate(self.layers):
                if self.do_res_norm:
                    residual = x
                if self.use_edge_attr:
                    x = layer(x, edge_index, edge_attr)
                else:
                    x = layer(x, edge_index)
                if self.do_res_norm:
                    x = self.norms[i](x + residual)
                x = self.leaky_relu(x)
                if self.training:
                    x = self.dropout(x)

        output = x

        pooling = self.pooling_layer(output, data.batch)

        if self.only_use_pooling:
            pooling = self.projection_layer(
                pooling
                # linear layer. possible alternative: why no projection in normal case (after pooling after GNN)?
            )

        if pooling.shape[0] == 1:
            graph_embedding = self.output(pooling)
        else:
            # pooling = einops.rearrange(pooling, "(bs ws) dim -> bs ws dim", bs=data.y.shape[0])
            graph_embedding = self.output(pooling)

        # print("mean graph embedding: ", torch.mean(abs(graph_embedding)))
        return graph_embedding
