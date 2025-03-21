import copy

import multiset
import torch

import wandb
from src.eval.retrieval_eval import eval_retrieval
from src.models.component_handler import similarity_from_config
from src.util.util import get_relation

from src.util.device import DEVICE


class RetrievalMetric:
    def __init__(self, config, model, dataset, log_prefix=""):
        self.dataset = dataset
        self.model = model
        self.log_prefix = log_prefix
        self.config = config

    def run(self):
        returndict = {}
        for retrieval_k in self.config.get("eval.retrieval.k", []):
            out = eval_retrieval(self.model, self.dataset,
                                 self.config, retrieval_k)
            for k, v in out.items():
                returndict[f"retrieval/{self.log_prefix}{k}"] = v
        return returndict


class SimilarityMetric:
    def __init__(self, config, model, dataset, log_prefix=""):
        self.dataset = [copy.deepcopy(d) for d in dataset]
        self.model = model
        self.sim = similarity_from_config(config)
        self.image_embeddings = torch.stack([d.y.to(DEVICE) for d in dataset])
        self.log_prefix = log_prefix

        img_to_img_sim_list = []
        for i in range(len(dataset)):
            for j in range(len(dataset)):
                img_to_img_sim_list.append(
                    self.sim(
                        self.image_embeddings[i].unsqueeze(0),
                        self.image_embeddings[j].unsqueeze(0),
                    ).item()
                )
        self.img_to_img = torch.tensor(img_to_img_sim_list).mean().item()
        wandb.log({f"{self.log_prefix}similarity_base": self.img_to_img}, step=0)

    def _graph_to_graph_mean_similarity(self):
        graph_embeddings = torch.stack(
            [self.model(d.to(DEVICE))[0] for d in self.dataset]
        )
        graph_to_graph_sim_list = []
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                graph_to_graph_sim_list.append(
                    self.sim(
                        graph_embeddings[i].unsqueeze(0),
                        graph_embeddings[j].unsqueeze(0),
                    ).item()
                )
        return torch.tensor(graph_to_graph_sim_list).mean().item()

    def run(self):
        return {
            f"{self.log_prefix}similarity_fit": self._graph_to_graph_mean_similarity()
            / self.img_to_img
        }


def build_prior_knowledge(data, config):
    """
    Create database (statistical prior knowledge).

    """
    sgg_modes = config.get("eval.sgg.modes", ["iter"])
    if (
        not sgg_modes
        or len(sgg_modes) == 0
        or config.get("eval.sgg.initial_edges", "") == "empty"
    ):
        print("no prior knowledge needed :)")
        return {}

    # distribution of relations over input, output node
    with_none = {}
    without_none = {}

    def incr(node_type_1, node_type_2, rel):
        if not (node_type_1, node_type_2) in with_none.keys():
            with_none[(node_type_1, node_type_2)] = {}
        distr_over_predicates = with_none[(node_type_1, node_type_2)]
        if rel not in distr_over_predicates.keys():
            distr_over_predicates[rel] = 0
        distr_over_predicates[rel] += 1
        if rel != None:
            if not (node_type_1, node_type_2) in without_none.keys():
                without_none[(node_type_1, node_type_2)] = {}
            distr_over_predicates = without_none[(node_type_1, node_type_2)]
            if rel not in distr_over_predicates.keys():
                distr_over_predicates[rel] = 0
            distr_over_predicates[rel] += 1

    for d in data:
        for i in range(len(d.x)):
            for j in range(len(d.x)):
                rel = get_relation(d, i, j)
                incr(d.node_human[i], d.node_human[j], rel)

    return {"rel_distr_with_none": with_none, "rel_distr_without_none": without_none}
