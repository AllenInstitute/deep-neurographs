"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Evaluates performance of edge classifier.

"""
from copy import deepcopy

import numpy as np


def run_evaluation(
    target_graphs, pred_graphs, y_pred, block_to_idxs, idx_to_edge, blocks
):
    stats = init_counters()
    stats_by_type = {"simple": init_counters(), "complex": init_counters()}
    for block_id in blocks:
        # Get predicted edges
        pred_edges = get_predictions(
            block_to_idxs[block_id], idx_to_edge, y_pred
        )

        # Overall performance
        num_fixes, num_mistakes = __reconstruction_stats(
            target_graphs[block_id], pred_graphs[block_id], pred_edges
        )
        stats["# splits fixed"].append(num_fixes)
        stats["# merges created"].append(num_mistakes)

        # In-depth performance
        simple_stats, complex_stats = __reconstruction_type_stats(
            target_graphs[block_id], pred_graphs[block_id], pred_edges
        )
        for key in stats.keys():
            stats_by_type["simple"][key].append(simple_stats[key])
            stats_by_type["complex"][key].append(complex_stats[key])
    return stats, stats_by_type


def init_counters(val=[]):
    return {"# splits fixed": deepcopy(val), "# merges created": deepcopy(val)}


def get_predictions(idxs, idx_to_edge, y_pred):
    edge_idxs = set(np.where(y_pred > 0)[0]).intersection(idxs)
    return set([idx_to_edge[idx] for idx in edge_idxs])


def __reconstruction_stats(target_graph, pred_graph, pred_edges):
    true_positives = 0
    false_positives = 0
    for edge in pred_edges:
        if edge in pred_graph.target_edges:
            true_positives += 1
        else:
            false_positives += 1
    return true_positives, false_positives


def __reconstruction_type_stats(target_graph, pred_graph, pred_edges):
    simple_stats = init_counters(val=0)
    complex_stats = init_counters(val=0)
    for edge in pred_edges:
        i, j = tuple(edge)
        deg_i = pred_graph.immutable_degree(i)
        deg_j = pred_graph.immutable_degree(j)
        if edge in pred_graph.target_edges:
            if deg_i == 1 and deg_j == 1:
                simple_stats["# splits fixed"] += 1
            else:
                complex_stats["# splits fixed"] += 1
        else:
            if deg_i == 1 and deg_j == 1:
                simple_stats["# merges created"] += 1
            else:
                complex_stats["# merges created"] += 1
    return simple_stats, complex_stats


def compute_accuracy(stats, type_key, num_edges):
    tp = deepcopy(stats[type_key]["# splits fixed"])
    fp = deepcopy(stats[type_key]["# merges created"])

    recall = tp / num_edges
    precision = tp / (tp + fp)
    f1 = (2 * recall * precision) / (recall + precision)
