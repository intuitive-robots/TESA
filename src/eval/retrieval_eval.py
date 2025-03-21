import torch
from torch_geometric.loader import DataLoader

from src.models.component_handler import logits_from_config

from src.util.device import DEVICE


def eval_retrieval(model, dataset_to_eval, config, retrieval_k):
    dl = DataLoader(
        dataset_to_eval, batch_size=retrieval_k, shuffle=True, drop_last=True
        # why shuffle? Because if adjacent graphs are similar, the task might be harder than intended.
    )
    scores = []
    for batch in dl:
        out = model(batch.to(DEVICE))
        y = batch.y.to(DEVICE).view_as(out)
        score = retrieval_from_to(out, y, config) / retrieval_k
        scores.append(score)
    return {f"/k={retrieval_k}": torch.mean(torch.tensor(scores))}


def retrieval_from_to(encodings1, encodings2, config):
    logits = logits_from_config(config)(encodings1, encodings2)[0]
    return sum(
        logits.argmax(dim=1)
        == torch.tensor(range(0, encodings1.shape[0]), device=encodings1.device)
    ).item()
