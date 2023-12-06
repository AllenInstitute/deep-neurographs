"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Evaluates performance of edge classifier.

"""
import numpy as np

STATS_LIST = [
    "precision",
    "recall",
    "f1",
    "# splits fixed",
    "# merges created",
]


def run_evaluation(
    target_graphs, pred_graphs, y_pred, block_to_idxs, idx_to_edge, blocks
):
    stats = dict([(s, []) for s in STATS_LIST])
    stats_by_type = {
        "simple": dict([(s, []) for s in STATS_LIST]),
        "complex": dict([(s, []) for s in STATS_LIST]),
    }
    print(blocks)
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
        if True:
            print("simple stats:", simple_stats)
            print("complex stats:", complex_stats)
            print("")
        for key in STATS_LIST:
            stats_by_type["simple"][key].append(simple_stats[key])
            stats_by_type["complex"][key].append(complex_stats[key])
    return stats, stats_by_type


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
    simple_stats = dict([(s, 0) for s in STATS_LIST])
    complex_stats = dict([(s, 0) for s in STATS_LIST])
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

    num_simple, num_complex = compute_edge_type(pred_graph)
    simple_stats = compute_accuracy(simple_stats, num_simple)
    complex_stats = compute_accuracy(complex_stats, num_complex)

    if False:
        print("# simple edges:", num_simple)
        print("% simple edges:", num_simple / (num_complex + num_simple))
        print("# complex edges:", num_complex)
        print("% complex edges:", num_complex / (num_complex + num_simple))
        print("")
    return simple_stats, complex_stats


def compute_edge_type(graph):
    num_simple = 0
    num_complex = 0
    for edge in graph.target_edges:
        i, j = tuple(edge)
        deg_i = graph.immutable_degree(i)
        deg_j = graph.immutable_degree(j)
        if deg_i == 1 and deg_j == 1:
            num_simple += 1
        else:
            num_complex += 1
    return num_simple, num_complex


def compute_accuracy(stats, num_edges):
    d = stats["# merges created"] + stats["# splits fixed"]
    r = 1 if num_edges == 0 else stats["# splits fixed"] / num_edges
    p = 1 if d == 0 else stats["# splits fixed"] / d
    stats["f1"] = 0 if r + p == 0 else (2 * r * p) / (r + p)
    stats["precision"] = p
    stats["recall"] = r
    return stats
