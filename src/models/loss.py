"""Loss functions for training the model. Convention: first argument is target (usually image features), second argument is prediction (usually graph features)."""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    r"""
    My own Contrastive Loss.
    """

    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, graph_features):
        r"""
        We minimize the distance between image and graph features by a contrastive loss.
        image_features and graph_features are of shape (batch_size, feature_size).
        """
        # calculate pairwise squared distances
        dists = torch.cdist(graph_features, image_features, p=2)
        dim = dists.shape[0]
        # shape: (graphs, imgs)
        pos_score = torch.exp(-dists / self.temperature)
        # remove diagonal
        total_loss = (torch.sum(pos_score) - torch.sum(torch.diag(pos_score))) / (
            dim * dim
        ) - torch.sum(torch.diag(pos_score)) / dim

        return total_loss


class CosAlignLoss(nn.Module):
    r"""
    "cosine". Not contrastive!
    Try to minimize error of cosine similarity between image and graph features.
    """

    def __init__(self):
        super(CosAlignLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, image_features, graph_features):

        return 1 - self.cos(image_features, graph_features).mean()


class CosCrossLoss(nn.Module):
    r"""
    Loss from CLIP.
    """

    def __init__(self):
        super(CosCrossLoss, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, image_features, graph_features):
        r"""
        image_features: (batch_size, feature_size)
        graph_features: (batch_size, feature_size)
        """
        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        graph_features = graph_features / graph_features.norm(dim=-1, keepdim=True)
        # cosine similarity
        cos_sim = torch.matmul(image_features, graph_features.T)
        return -self.softmax(cos_sim).diag().log().mean()


class WinoLoss(nn.Module):
    r"""
    Contrastive Hinge Loss. Using Cos-Similarity.
    "Push away closest negative" and "Pull closer positive".

    Modified, from Yufeng Huang (MIT License) - StructureCLIP.
    """

    def __init__(self, margin):
        super(WinoLoss, self).__init__()
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction="mean")
        self.relu = nn.ReLU()
        self.hard_counter = 0  # first 100 calls soft.

    def forward(self, image_features, graph_features):
        self.hard_counter += 1
        is_hard = self.hard_counter >= 100
        # last
        batch_size = image_features.shape[0]
        cos_img2text = torch.matmul(image_features, graph_features.T)  # [bs,bs]

        pos_score = torch.diag(cos_img2text)  # [bs]
        img_neg_score = torch.max(
            cos_img2text - 10 * torch.eye(batch_size, requires_grad=False).cuda(0),
            dim=-1,
        )[
            0
        ]  # [bs]

        cos_text2img = cos_img2text.T  # [bs,bs]
        if is_hard:
            text_neg_score = torch.max(
                cos_text2img - 10 * torch.eye(batch_size, requires_grad=False).cuda(0),
                dim=-1,
            )[
                0
            ]  # [bs]
        else:
            text_neg_score = torch.mean(cos_text2img, dim=-1)
        # text_neg_score = torch.max(cos_text2img - 10 * torch.eye(batch_size, requires_grad=False).cuda(0), dim=-1)[0] # [bs]
        margin = torch.ones_like(pos_score, requires_grad=False) * 0.2  # [bs]

        loss = self.relu(img_neg_score + margin - pos_score) + self.relu(
            text_neg_score + margin - pos_score
        )
        loss = torch.mean(loss)
        return loss


ContrastiveCosLoss = WinoLoss


class OwnMSESimilarity(nn.Module):
    def forward(self, image_features, graph_features):
        # image features of shape [batch_size, feature_size]
        # graph_features of shape [1, feature_size]
        # take sigmoid of -MSE
        graph_features = graph_features.expand(image_features.shape)
        mse = torch.nn.functional.mse_loss(
            image_features, graph_features, reduction="none"
        ).mean(dim=1)
        return 2 * torch.sigmoid(-mse)  # result in [0,1]
