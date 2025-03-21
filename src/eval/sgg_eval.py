import itertools
import random

import torch
from multiset import Multiset
from tqdm import tqdm

from src.eval.scene_graph_generation import (build_initial_edges,
                                             correct_prediction,
                                             iterative_probabilities,
                                             pairwise_probabilities,
                                             pass_initial_probabilities,
                                             refined_probabilities)
from src.models.component_handler import similarity_from_config
from src.util.util import filtergraphs, get_relation


def object_groundtruth_rows(d):
    r"""
    Output: ("subjectType", "objectType", predicate string)
    """
    out = []
    for i, j, rel in d.edge_human:
        out.append((d.node_human[i], d.node_human[j], rel))
    return list(out)


def recall_sgg(object_prob_rows, d, recall_at):
    """

    @param object_prob_rows: list of ("subjectType", "objectType", predicate string) sorted descending after probabilities.

    Runs on single data object. Recall is in this case multi-class micro-averaged recall.
    """
    predict_rows = object_prob_rows[:recall_at]  # keep first R entries
    correct_amt = 0
    ground_truth_rows = object_groundtruth_rows(d)  # no None's here.
    total = len(ground_truth_rows)
    for r in predict_rows:
        if r in ground_truth_rows:
            correct_amt += 1
            ground_truth_rows.remove(
                r
            )  # removes the triplet once from the ground truth.
    return correct_amt / total


def triple_rows(prob_rows, d, filter_nones=True, constrained=False):
    r"""
    Input prob_quadruples:  list of (probability, from_node index, to_node index, predicate string)
    Output: list of ("subjectType", "objectType", predicate string), sorted descending after probability.
    """

    def map_quadruple_to_triple(quadruple):
        (_, from_idx, to_idx, predicate) = quadruple
        return (d.node_human[from_idx], d.node_human[to_idx], predicate)

    # filter out nones:
    if filter_nones:
        prob_rows = list(filter(lambda x: x[3] is not None, prob_rows))
    # sort by probability:
    prob_rows.sort(reverse=True, key=lambda x: (x[0], random.random()))
    if constrained:
        # take only prob_row with higest probability for each edge.
        # filter out all elements repeating a previous (from_node, to_node, predicate)-combination:
        seen = set()
        filtered_prob_rows = []
        for item in prob_rows:
            key = (item[1], item[2])  # (subjectID, objectID, predicate)
            if key not in seen:
                seen.add(key)
                filtered_prob_rows.append(item)
    else:
        filtered_prob_rows = prob_rows
    out = list(map(map_quadruple_to_triple, filtered_prob_rows))
    return out


def node_prob_rows(
    g_model,
    d,
    edges_forward_backward,
    similarity_fn,
    sgg_mode,
    include_none,
    initial_edges,
):
    r"""
    returns list of (probability, from_node_idx, to_node_idx, predicate) tuples.
    """
    with torch.no_grad():
        g_model.eval()
        if sgg_mode == "initial":
            prob_quadruples_node_based = pass_initial_probabilities(
                initial_edges)
        elif sgg_mode == "pair":
            prob_quadruples_node_based = pairwise_probabilities(
                g_model,
                d.x,
                edges_forward_backward,
                d.y,
                similarity_fn,
                initial_edges,
                include_none,
            )
        elif sgg_mode == "iter":
            prob_quadruples_node_based = iterative_probabilities(
                g_model,
                d.x,
                edges_forward_backward,
                d.y,
                similarity_fn,
                include_none,
                initial_edges,
            )
        elif sgg_mode.startswith("refined"):
            refinement_level = int(sgg_mode.replace("refined", ""))
            prob_quadruples_node_based = refined_probabilities(
                g_model,
                d.x,
                edges_forward_backward,
                d.y,
                similarity_fn,
                include_none,
                refinement_level,
                initial_edges,
            )
        else:
            raise ValueError(f"sgg_mode {sgg_mode} not supported")

    return list(prob_quadruples_node_based)


def link_acc_sgg(
    all_rows,
    d,
):
    """
    runs on single data object.
    """
    with torch.no_grad():
        # all_rows: (probability, from_node_idx, to_node_idx, predicate)

        # link accuracy.
        correct_amt = 0
        all_rows.sort(key=lambda x: (x[1], x[2]))
        for (from_idx, to_idx), group in itertools.groupby(
            all_rows, key=lambda x: (x[1], x[2])
        ):
            max_tuple = max(group, key=lambda x: x[0])
            prediction = max_tuple[3]
            real = get_relation(d, from_idx, to_idx)
            if correct_prediction(prediction, real):
                correct_amt += 1

        actual_amt = len(d.x) * (len(d.x) - 1)
        return correct_amt / actual_amt


def recall_sgg_per_predicate(
    object_prob_rows,
    d,
    recall_at,
    correct_predicates: Multiset,
    total_predicates: Multiset,
):
    """
    @param object_prob_rows: list of ("subjectType", "objectType", predicate string) sorted descending after probabilities.
    runs on single data object. Recall is in this case multi-class macro-averaged recall.

    @param correct_guesses, total_predicates are tracking the number of guesses.
    """
    predict_rows = object_prob_rows[:recall_at]  # keep first R entries

    ground_truth_rows = object_groundtruth_rows(d)  # no None's here.
    for r in ground_truth_rows:
        total_predicates.add(r[2])
        if r in predict_rows:
            correct_predicates.add(r[2])


def eval_sgg(
    encodings_fb,
    g_model,
    dataset,
    config,
    sgg_mode,
    prior_knowledge,
    print_predictions,
):
    (dataset_to_eval, dataset_to_skip, do_lowerbound) = filtergraphs(
        dataset, config, "eval_SGG"
    )
    with torch.no_grad():
        recalls = config.get("eval.sgg.recalls", [])
        metrics = config.get("eval.sgg.metrics", [])
        do_link_acc = "link_acc" in metrics
        do_recall = "recall" in metrics
        do_mean_recall = "mean_recall" in metrics
        constrain = config.get("eval.sgg.constrain", [])
        do_unconstrained = "unconstrained" in constrain
        do_constrained = "constrained" in constrain

        do_fast_estimate = len(dataset_to_skip) > 0

        sim_fn = similarity_from_config(config)

        est_prefix = "estimate-" if do_fast_estimate else ""
        lb_prefix = "lower_bound-"

        # collect from graphs:
        if do_link_acc:
            link_acc_per_graph = []
        if do_unconstrained:
            recall_per_graph_unconstrained = {r: [] for r in recalls}
            total_rows_per_predicate_unconstrained = {
                r: Multiset() for r in recalls}
            correct_rows_per_predicate_unconstrained = {
                r: Multiset() for r in recalls}
        if do_constrained:
            recall_per_graph_constrained = {r: [] for r in recalls}
            total_rows_per_predicate_constrained = {
                r: Multiset() for r in recalls}
            correct_rows_per_predicate_constrained = {
                r: Multiset() for r in recalls}

        for d in tqdm(dataset_to_eval, f"eval: {sgg_mode}-sgg"):
            initial_edges = build_initial_edges(
                d, config["eval.sgg.initial_edges"], prior_knowledge
            )
            prob_rows = node_prob_rows(
                g_model,
                d,
                encodings_fb,
                sim_fn,
                sgg_mode,
                True,
                initial_edges,
            )
            if do_link_acc:
                link_acc_per_graph.append(link_acc_sgg(prob_rows, d))

            if do_unconstrained:
                guess_triples_unconstrained = triple_rows(
                    prob_rows, d, filter_nones=True, constrained=False
                )
                for recall_at in recalls:
                    recall_per_graph_unconstrained[recall_at].append(
                        recall_sgg(guess_triples_unconstrained, d, recall_at)
                    )
                    recall_sgg_per_predicate(
                        guess_triples_unconstrained,
                        d,
                        recall_at,
                        correct_rows_per_predicate_unconstrained[recall_at],
                        total_rows_per_predicate_unconstrained[recall_at],
                    )
            if do_constrained:
                guess_triples_constrained = triple_rows(
                    prob_rows, d, filter_nones=True, constrained=True
                )
                for recall_at in recalls:
                    recall_per_graph_constrained[recall_at].append(
                        recall_sgg(guess_triples_constrained, d, recall_at)
                    )
                    recall_sgg_per_predicate(
                        guess_triples_constrained,
                        d,
                        recall_at,
                        correct_rows_per_predicate_constrained[recall_at],
                        total_rows_per_predicate_constrained[recall_at],
                    )
            if print_predictions:
                print(f"for graph [nodes = {len(d.x)}]:", d.node_human)
                print(f"Groundtruth: {object_groundtruth_rows(d)}")
                if do_unconstrained:
                    print("Unconstrained Predictions@20:")
                    print(guess_triples_unconstrained[:20])
                if do_constrained:
                    print("Constrained Predictions@20:")
                    print(guess_triples_constrained[:20])

        def calculate_mean_recall(correct_rows_per_predicate, total_rows_per_predicate):
            return torch.mean(
                torch.tensor(
                    [
                        correct_rows_per_predicate.get(predicate, 0)
                        / total_rows_per_predicate.get(predicate, 0)
                        for predicate in total_rows_per_predicate.distinct_elements()
                    ]
                )
            )

        # instead of dividing by length of dataset we want to devide by length of hypothetical dataset including dropped elements.
        zero_inflate_factor = len(dataset_to_eval) / (
            len(dataset_to_eval) + len(dataset_to_skip)
        )

        returndict = {}
        if do_link_acc:
            acc = torch.mean(torch.tensor(link_acc_per_graph))
            returndict[f"/{sgg_mode}/{est_prefix}sgg-link-acc"] = acc
            if do_lowerbound:
                returndict[f"/{sgg_mode}/{lb_prefix}sgg-link-acc"] = (
                    acc * zero_inflate_factor
                )

        def add_recall_values_to_returndict(
            uc_or_c_prefix,
            recall_per_graph,
            correct_rows_per_predicate,
            total_rows_per_predicate,
            returndict,
        ):
            for recall_at in recalls:
                if do_recall:
                    recall_val = torch.mean(
                        torch.tensor(recall_per_graph[recall_at]))
                    returndict[
                        f"/{uc_or_c_prefix}/{sgg_mode}/{est_prefix}R@{recall_at}"
                    ] = recall_val
                if do_mean_recall:
                    returndict[
                        f"/{uc_or_c_prefix}/{sgg_mode}/{est_prefix}mR@{recall_at}"
                    ] = calculate_mean_recall(
                        correct_rows_per_predicate[recall_at],
                        total_rows_per_predicate[recall_at],
                    )
            if do_lowerbound:
                # increment total rows per-predicate
                for recall_at in recalls:
                    for d in dataset_to_skip:
                        recall_sgg_per_predicate(
                            [],
                            d,
                            recall_at,
                            # results for correct should not change.
                            Multiset(),
                            total_rows_per_predicate[recall_at],
                        )

                for recall_at in recalls:
                    if do_recall:
                        returndict[
                            f"/{uc_or_c_prefix}/{sgg_mode}/{lb_prefix}R@{recall_at}"
                        ] = (
                            torch.mean(torch.tensor(
                                recall_per_graph[recall_at]))
                            * zero_inflate_factor
                        )
                    if do_mean_recall:
                        returndict[
                            f"/{uc_or_c_prefix}/{sgg_mode}/{lb_prefix}mR@{recall_at}"
                        ] = calculate_mean_recall(
                            correct_rows_per_predicate[recall_at],
                            total_rows_per_predicate[recall_at],
                        )

        if do_unconstrained:
            add_recall_values_to_returndict(
                "UC",
                recall_per_graph_unconstrained,
                correct_rows_per_predicate_unconstrained,
                total_rows_per_predicate_unconstrained,
                returndict,
            )
        if do_constrained:
            add_recall_values_to_returndict(
                "C",
                recall_per_graph_constrained,
                correct_rows_per_predicate_constrained,
                total_rows_per_predicate_constrained,
                returndict,
            )
        return returndict
