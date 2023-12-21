"""
Created on Sat Dec 12 17:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org



"""

import numpy as np


def get_reconstructions(
    pred_neurographs,
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
        # Get positive edge predictions
        edge_probs = get_edge_probs(
            idx_to_edge, y_pred, low_threshold, valid_idxs=block_to_idxs[block_id]
        )

        # Refine predictions wrt structure
        if structure_aware:
            edge_preds[block_id] = get_structure_aware_prediction(
                pred_neurographs[block_id],
                edge_probs,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
        else:
            edge_preds[block_id] = edge_probs.keys()
    return edge_preds


def get_reconstruction(
    pred_neurograph,
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
            pred_neurograph,
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
    pred_neurograph, edge_probs, high_threshold=0.8, low_threshold=0.6
):
    # Initializations
    edge_preds = list(edge_probs.keys())
    pred_neurograph.init_predicted_graph()

    # Add best simple edges
    visited_nodes = set()
    remaining_edge_preds = []
    viable_edge_preds = []
    dists = [pred_neurograph.compute_length(edge) for edge in edge_preds]
    for idx in np.argsort(dists):
        edge = edge_preds[idx]
        if pred_neurograph.is_simple(edge) and edge_probs[edge] > high_threshold:
            if not pred_neurograph.creates_cycle(tuple(edge)):
                viable_edge_preds.append(edge)
        else:
            remaining_edge_preds.append(edge)

    # Add remaining valid edges
    for edge in remaining_edge_preds:
        if not pred_neurograph.creates_cycle(tuple(edge)):
            viable_edge_preds.append(edge)
    return viable_edge_preds
