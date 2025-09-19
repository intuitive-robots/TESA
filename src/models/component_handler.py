import torch.nn as nn
import torch.nn.functional as F

from src.eval.qa_loader import (gqa_load_collection,
                                vqa_load_collection)
from src.models.gcn import GCN, MixedN, trivial_base
from src.models.graph_embedder import GraphEmbedder
from src.models.loss import (ContrastiveCosLoss, ContrastiveLoss, CosAlignLoss,
                             CosCrossLoss, OwnMSESimilarity)
from src.util.util import cosine_similarity_logits, own_mse_logits
from src.util.device import DEVICE


def model_from_config(config, input_dim, output_dim, edge_attr_dim=None):

    architecture = config.get("architecture", "GraphEmbedder")
    hidden_dim = config.get("hidden_dim", 512)
    if architecture == "GraphEmbedder":
        (num_layer, layer_type, pool_type, heads, dropout) = (
            config.get(a)
            for a in ("num_layer", "layer_type", "pool_type", "heads", "dropout")
        )
        model = GraphEmbedder(
            input_dim,
            hidden_dim,
            output_dim,
            num_layer,
            layer_type,
            pool_type,
            heads,
            dropout,
            edge_dim=edge_attr_dim,  # 1 for edge direction
            do_res_norm=config.get("layer_type.res_norm", False),
        ).to(DEVICE)
    elif architecture == "GCN":
        model = GCN(input_dim, hidden_dim, output_dim).to(DEVICE)
    elif architecture == "trivial":
        model = trivial_base(input_dim, output_dim)
    elif architecture == "MixedN":
        model = MixedN(input_dim, 12, output_dim).to(DEVICE)
    else:
        raise ValueError(f"architecture {architecture} not supported")
    return model


def get_loss(loss, config):
    if loss == "None":
        return lambda _, __: 0
    # non-contrastive
    if loss == "MSE":
        return F.mse_loss
    if loss == "MAE":
        return F.l1_loss
    if loss == "cos-align":
        return CosAlignLoss()
    # contrastive
    if loss == "cos-margin":
        return ContrastiveCosLoss(config.get("margin", 0.1))
    if loss == "cos-cross":
        return CosCrossLoss()

    raise ValueError(f"loss {loss} not supported")


def loss_from_config(config):
    loss = config["loss"]
    return get_loss(loss, config)


def similarity_from_config(config):
    assert config["loss"] in ["cos-align", "cos-margin", "cos-cross", "MSE"]
    if config["loss"] == "MSE":
        return OwnMSESimilarity()
    return nn.CosineSimilarity(dim=1, eps=1e-6)


def logits_from_config(config):
    if config["loss"] in ["cos-align", "cos-margin", "cos-cross"]:
        return cosine_similarity_logits
    if config["loss"] == "MSE":
        return own_mse_logits


def contrastive_loss_from_config(config):
    contrastive = config.get("contrastive_loss", False)
    if not contrastive:
        return None
    if contrastive == True:
        assert config["loss"] == "MSE"
        return ContrastiveLoss()  # selfmade euclidean contrastive loss,
    return get_loss(contrastive, config)


def qa_collection(config, answers=None):
    if config["dataset"] == "gqa":
        return gqa_load_collection(config["eval.qa.gqa.question_file"], answers)
    if config["dataset"] == "psg":
        return vqa_load_collection()
    else:
        raise ValueError(
            f"dataset {config['dataset']} does not support question collections."
        )
