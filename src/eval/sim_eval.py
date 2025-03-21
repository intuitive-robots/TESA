import torch
from tqdm import tqdm

from src.models.component_handler import similarity_from_config

from src.util.device import DEVICE


def eval_sim(test_data, config, model):
    r"""Evaluate similarity between graph and image embedding."""
    sim = similarity_from_config(config)
    sims = []
    with torch.no_grad():
        model.eval()
        for g in tqdm(test_data, desc="sim eval"):
            g = g.to(DEVICE)
            out = model(g)[0]
            y = g.y
            sims.append(sim(out.unsqueeze(0), y.unsqueeze(0)).item())
    return {"/mean": sum(sims) / len(sims)}
