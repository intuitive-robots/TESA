import torch.nn as nn
from torch_geometric.nn import (FiLMConv, GATConv, GATv2Conv, GCNConv, GINConv,
                                TransformerConv)


def layer_factory(layer_type, in_dim, out_dim, heads, dropout, edge_dim):
    if layer_type == "GAT":
        return GATConv(
            in_dim,
            out_dim,
            concat=False,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
    if layer_type == "GATv2":
        return GATv2Conv(
            in_dim,
            out_dim,
            concat=False,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
    if layer_type == "TransConv":
        return TransformerConv(
            in_dim,
            out_dim,
            heads=heads,
            dropout=dropout,
            concat=False,
            edge_dim=edge_dim,
        )
    if layer_type == "GCN":
        flat_layer_class = GCNConv
    elif layer_type == "GIN":
        return GINConv(nn=nn.Linear(in_dim, out_dim), eps=1, train_eps=True)
    elif layer_type == "FiLM":
        flat_layer_class = FiLMConv
    else:
        raise ValueError(f"layer_type {layer_type} not supported")
    return flat_layer_class(in_dim, out_dim)
