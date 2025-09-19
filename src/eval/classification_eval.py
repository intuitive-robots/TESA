from torch_geometric.loader import DataLoader

from src.util.device import DEVICE
from src.util.util_siglip import siglip_encode_textList
from src.util.util_clip import clip_encode_text_list
import torch
import torch.nn.functional as F

from tqdm import tqdm

def eval_classification(model, dataset_to_eval, config, bs):
    data = {}
    for p in dataset_to_eval:
        if p.label not in data:
            data[p.label] = p.description
    ground_truth = []
    for i in range(80):
        if i == 70:
            ground_truth.append("In this photo you can see a toaster.")
        else:
            ground_truth.append(data[i])
    
    dl = DataLoader(
        dataset_to_eval, batch_size=bs, shuffle=True, drop_last=True
        # why shuffle? Because if adjacent graphs are similar, the task might be harder than intended.
    )
    if config["dataset.base_embedding.using"] == "SigLIP":
        gt_encodings = siglip_encode_textList(ground_truth)
    elif config["dataset.base_embedding.using"] == "clip":
        gt_encodings = clip_encode_text_list(ground_truth).to(torch.float32)
    else:
        raise Exception("Model not defined!")
    ground_truth.append("")
    gt_to_idx = {desc: idx for idx, desc in enumerate(ground_truth)}
    
    num_correct_graph = 0
    num_correct_img = 0
    all_points = 0
    for batch in tqdm(dl):
        graph_embeddings = model(batch.to(DEVICE))
        img_embeddings = batch.y.to(DEVICE).view_as(graph_embeddings)

        graph_norm = F.normalize(graph_embeddings, p=2, dim=1)
        image_norm = F.normalize(img_embeddings, p=2, dim=1)
        gt_norm = F.normalize(gt_encodings, p=2, dim=1)

        similarities_graph = torch.matmul(graph_norm, gt_norm.T)
        similarities_img = torch.matmul(image_norm, gt_norm.T)

        best_similarities_graph, best_indices_graph = similarities_graph.max(dim=1)
        best_similarities_img, best_indices_img = similarities_img.max(dim=1)

        batch_indices = torch.tensor([gt_to_idx[desc] for desc in batch.description]).to(best_indices_graph.device)

        graph_mask = best_indices_graph == batch_indices
        img_mask = best_indices_img == batch_indices

        num_correct_graph += graph_mask.sum().item()
        num_correct_img += img_mask.sum().item()

        all_points += len(batch.ptr) - 1
    
    return num_correct_graph / all_points, num_correct_img / all_points