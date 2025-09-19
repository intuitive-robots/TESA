from torch_geometric.loader import DataLoader

from src.models.component_handler import logits_from_config

from src.util.device import DEVICE

from tqdm import tqdm

import torch
import torch.nn.functional as F

def eval_image_captioning(model, dataset_to_eval, config, retrieval_k, recall_k):
    
    dl_img = DataLoader(dataset_to_eval[0], batch_size=retrieval_k, shuffle=False, drop_last=True)
    dl_text = DataLoader(dataset_to_eval[1], batch_size=retrieval_k, shuffle=False, drop_last=True)
    dl_graph = DataLoader(dataset_to_eval[2], batch_size=retrieval_k, shuffle=False, drop_last=True)
    
    correct_graph = 0
    correct_img = 0
    total = 0

    for b_img, b_text, b_graph in tqdm(zip(dl_img, dl_text, dl_graph)):

        similarities_graph = torch.matmul(b_graph, b_text.T)   # [B, B]
        similarities_img   = torch.matmul(b_img, b_text.T)     # [B, B]

        batch_size = b_text.size(0)
        gt = torch.arange(batch_size, device=b_text.device)

        # Top-k indices
        topk_graph = similarities_graph.topk(recall_k, dim=1).indices  # [B, K]
        topk_img   = similarities_img.topk(recall_k, dim=1).indices    # [B, K]

        # Check if ground-truth index is within top-k
        mask_graph = (topk_graph == gt.unsqueeze(1)).any(dim=1)
        mask_img   = (topk_img == gt.unsqueeze(1)).any(dim=1)

        correct_graph += mask_graph.sum().item()
        correct_img   += mask_img.sum().item()
        total += batch_size

    return correct_graph / total, correct_img / total


