import random

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.util.device import DEVICE


def make_candidate_graphs_for_edge(
    nodes,
    edges_forward_backward,
    candidate_predicates,
    from_node_idx,
    to_node_idx,
    # List: (from_node_idx, to_node_idx, predicate)
    default_edge_predicates=None,
):
    """
    Makes a list of hypergraphs for all possible predicates between two nodes.
    These hypergraphs are all identical except for the connection between the two nodes, which will have all values corresponding to
    edges_forward_backward and also None.
    """

    # modification-candidate edge is "first edge".
    # 1. build edge index
    edge_index_list_missing_first = []
    if default_edge_predicates is None:
        default_edge_predicates = []
    for i, j, _ in default_edge_predicates:
        if not (i == from_node_idx and j == to_node_idx):
            edge_index_list_missing_first.append([i, j])
            edge_index_list_missing_first.append([j, i])
    first_edge_idx = [
        [from_node_idx, to_node_idx],
        [to_node_idx, from_node_idx],
    ]
    edge_index_including_first = (
        torch.tensor(first_edge_idx +
                     edge_index_list_missing_first).t().contiguous()
    )

    # 2. build edge_attr, without the first edge.
    edge_attr_list_missing_first = []
    for i, j, predicate in default_edge_predicates:
        if not (i == from_node_idx and j == to_node_idx):
            edge_attr_f, edge_attr_b = edges_forward_backward[predicate]
            edge_attr_list_missing_first.append(edge_attr_f)
            edge_attr_list_missing_first.append(edge_attr_b)
    if len(edge_attr_list_missing_first) != 0:
        edge_attr_missing_first = torch.cat(edge_attr_list_missing_first)
    else:
        edge_attr_missing_first = torch.empty((0))
    # build graphs that modify the edge between from_node_idx and to_node_idx.
    for predicate in candidate_predicates:
        if predicate is not None:
            edge_attr_f, edge_attr_b = edges_forward_backward[predicate]
            first_edge_attr = torch.cat([edge_attr_f, edge_attr_b])
            edge_attr = torch.cat([first_edge_attr, edge_attr_missing_first])
            edge_index = edge_index_including_first
        else:  # none-predicate:
            edge_attr = edge_attr_missing_first
            if len(edge_index_list_missing_first) != 0:
                edge_index = (
                    torch.tensor(
                        edge_index_list_missing_first).t().contiguous()
                )
            else:
                edge_index = torch.empty((2, 0))

        yield Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr)
        # yield HypergraphData(
        #     x=nodes,
        #     edge_index=edge_index,
        #     edge_attr=edge_attr,
        #     hyperedge_index_nodes=torch.tensor([]),
        #     hyperedge_index_hyperedges=torch.tensor([]),
        # )


def iterative_probabilities(
    g_model,
    nodes,
    predicate_to_encodings,
    target_y,
    similarity_fn,
    include_none,
    initial_edges,
):
    # 1. add all edges iteratively.
    if include_none:
        candidate_predicates = list(predicate_to_encodings.keys()) + [None]
    else:
        candidate_predicates = list(predicate_to_encodings.keys())
    # List: (from_node_idx, to_node_idx, predicate)
    current_edges = initial_edges
    for i, _ in enumerate(nodes):
        for j, _ in enumerate(nodes):
            if i == j:
                continue
            # update edge from i to j:
            probs = probabilities_for_edge(
                g_model,
                nodes,
                predicate_to_encodings,
                target_y,
                similarity_fn,
                candidate_predicates,
                i,
                j,
                current_edges,
            )
            max_idx = torch.argmax(probs)
            if candidate_predicates[max_idx] is not None:
                current_edges.append((i, j, candidate_predicates[max_idx]))
    # 2. run prediction
    all_rows = list(
        pairwise_probabilities(
            g_model,
            nodes,
            predicate_to_encodings,
            target_y,
            similarity_fn,
            current_edges,
            include_none,
        )
    )
    return all_rows


def refined_probabilities(
    g_model,
    nodes,
    predicate_to_encodings,
    target_y,
    similarity_fn,
    include_none,
    number_of_iterations,
    initial_edges,
):
    next_edges = initial_edges  # List: (from_node_idx, to_node_idx, predicate)
    # always include none during non-last iterations!
    candidate_predicates = list(predicate_to_encodings.keys()) + [None]

    for i in range(number_of_iterations - 1):
        # from basis of old edge predictions, re-predict edges (-> next_edges)
        # List: (from_node_idx, to_node_idx, predicate)
        last_edges = next_edges
        next_edges = []
        for i, _ in enumerate(nodes):
            for j, _ in enumerate(nodes):
                if i == j:
                    continue
                # update edge from i to j:
                probs = probabilities_for_edge(
                    g_model,
                    nodes,
                    predicate_to_encodings,
                    target_y,
                    similarity_fn,
                    candidate_predicates,
                    i,
                    j,
                    last_edges,
                )
                max_idx = torch.argmax(probs)
                if candidate_predicates[max_idx] is not None:
                    next_edges.append((i, j, candidate_predicates[max_idx]))

    return pairwise_probabilities(
        g_model,
        nodes,
        predicate_to_encodings,
        target_y,
        similarity_fn,
        next_edges,
        include_none,
    )


def build_initial_edges(d, initial_edges_config, prior_knowledge):
    if initial_edges_config == "empty":
        return []
    elif initial_edges_config in ["random_with_none", "random_without_none"]:
        edges = []
        if initial_edges_config == "random_without_none":
            distribution = prior_knowledge["rel_distr_without_none"]
        else:
            distribution = prior_knowledge["rel_distr_with_none"]
        # sample edges
        for i in range(len(d.x)):
            for j in range(len(d.x)):
                if i == j:
                    continue
                else:
                    edge_distr = distribution.get(
                        (d.node_human[i], d.node_human[j]), None
                    )
                    if edge_distr == None or len(edge_distr) == 0:
                        continue
                    selected = random.choices(
                        list(edge_distr.keys()), weights=edge_distr.values(), k=1
                    )[0]
                    if selected is not None:
                        edges.append((i, j, selected))
        return edges
    else:
        raise ValueError(
            f"Unknown initial_edges_config: {initial_edges_config}")


def pass_initial_probabilities(
    default_edge_predicates,
):
    for from_idx, to_idx, predicate in default_edge_predicates:
        yield (
            1,
            from_idx,
            to_idx,
            predicate,
        )


def pairwise_probabilities(
    g_model,
    nodes,
    predicate_to_encodings,
    target_y,
    similarity_fn,
    default_edge_predicates,
    include_none,
):
    """
    @param g_model: the model to evaluate
    @param nodes: a list of node embeddings
    @param candidate_predicates: a dict predicate -> (forward_embedding, backward_embedding)
    @param target_y: the target output
    @param similarity_fn: a function that takes two outputs of the model and returns a similarity score (higher is better)
    @param default_edge_predicates: a list of (from_node_idx, to_node_idx, predicate) tuples that are always included in the graph (except when overwritten by the to-be-predicted edge)

    Evaluates the model on all possible pairs of nodes in the graph for all possible predicates.
    Returns a list of (probability, from_node, to_node, predicate) tuples.
    """
    target_y = target_y.to(DEVICE)

    if include_none:
        candidate_predicates = list(predicate_to_encodings.keys()) + [None]
    else:
        candidate_predicates = list(predicate_to_encodings.keys())
    for i, _ in enumerate(nodes):
        for j, _ in enumerate(nodes):
            if i == j:
                continue
            probs = probabilities_for_edge(
                g_model,
                nodes,
                predicate_to_encodings,
                target_y,
                similarity_fn,
                candidate_predicates,
                i,
                j,
                default_edge_predicates,
            )
            for p_idx, predicate in enumerate(candidate_predicates):
                yield (
                    probs[p_idx].item(),
                    i,
                    j,
                    predicate,
                )


def probabilities_for_edge(
    g_model,
    nodes,
    predicate_to_encodings,
    target_y,
    similarity_fn,
    candidate_predicates,
    i,
    j,
    default_edge_predicates=None,
):
    r"""
    @param default_edge_predicates:  List (from_node_idx, to_node_idx, predicate)
    """
    hypergraphs = make_candidate_graphs_for_edge(
        nodes,
        predicate_to_encodings,
        candidate_predicates,
        i,
        j,
        default_edge_predicates,
    )
    loader = DataLoader(list(hypergraphs), batch_size=1000)
    candidate_y = []
    for batch in loader:
        candidate_y.append(g_model(batch.to(DEVICE)))
    # shape (num_candidates, embedding_size)
    candidate_y = torch.cat(candidate_y)
    candidate_similarity = similarity_fn(
        candidate_y, target_y.unsqueeze(0).to(DEVICE))
    probs = torch.nn.functional.softmax(candidate_similarity, dim=0)
    return probs


def correct_prediction(prediction, relation):
    if (not relation) or (not prediction):  # one of them is None.
        return (not relation) and (not prediction)
    else:
        return relation == prediction
