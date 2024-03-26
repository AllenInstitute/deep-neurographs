"""
Created on Sat Dec 12 17:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org



"""

import numpy as np

from deep_neurographs import graph_utils as gutils


def get_reconstructions(
    neurographs,
    blocks,
    block_to_idxs,
    idx_to_edge,
    y_pred,
    high_threshold=0.9,
    low_threshold=0.6,
    structure_aware=True,
):
    edge_preds = dict()
    for block_id in blocks:
        # Get positive predictions
        edge_probs = get_edge_probs(
            idx_to_edge,
            y_pred,
            low_threshold,
            valid_idxs=block_to_idxs[block_id],
        )

        # Refine predictions wrt structure
        if structure_aware:
            edge_preds[block_id] = get_structure_aware_prediction(
                neurographs[block_id],
                edge_probs,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
        else:
            edge_preds[block_id] = edge_probs.keys()
    return edge_preds


def get_reconstruction(
    neurograph,
    y_pred,
    idx_to_edge,
    high_threshold=0.9,
    low_threshold=0.6,
    structure_aware=True,
):
    # Get positive edge predictions
    edge_probs = get_edge_probs(idx_to_edge, y_pred, low_threshold)
    if structure_aware:
        return get_structure_aware_prediction(
            neurograph,
            edge_probs,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
    else:
        return edge_probs.keys()


def get_edge_probs(idx_to_edge, y_pred, threshold, valid_idxs=[]):
    edge_probs = dict()
    for i, y_i in enumerate(y_pred):
        contained_bool = True if len(valid_idxs) == 0 else i in valid_idxs
        if y_i > threshold and contained_bool:
            edge_probs[idx_to_edge[i]] = y_i
    return edge_probs


def get_structure_aware_prediction(
    neurograph, probs, high_threshold=0.8, low_threshold=0.6
):
    # Initializations
    proposals = list(probs.keys())
    pred_graph = neurograph.copy_graph()

    # Add best simple edges
    positive_predictions = []
    remaining_proposals = []
    dists = [neurograph.proposal_length(edge) for edge in proposals]
    # dists = [probs[edge] for edge in proposals]
    for idx in np.argsort(dists):
        edge = proposals[idx]
        if neurograph.is_simple(edge) and probs[edge] > high_threshold:
            if not gutils.creates_cycle(pred_graph, tuple(edge)):
                pred_graph.add_edges_from([edge])
                positive_predictions.append(edge)
        else:
            remaining_proposals.append(edge)

    # Add remaining viable edges
    for edge in remaining_proposals:
        if not gutils.creates_cycle(pred_graph, tuple(edge)):
            pred_graph.add_edges_from([edge])
            positive_predictions.append(edge)
    return positive_predictions
