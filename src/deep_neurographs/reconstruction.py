"""
Created on Sat Dec 12 17:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

...

"""

import os
from random import sample

import numpy as np

from deep_neurographs import graph_utils as gutils
from deep_neurographs import swc_utils, utils


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
    preds = []
    remaining_proposals = []
    dists = [neurograph.proposal_length(edge) for edge in proposals]
    #dists = [probs[edge] for edge in proposals]
    for idx in np.argsort(dists):
        edge = proposals[idx]
        if neurograph.is_simple(edge) and probs[edge] > high_threshold:
            if not gutils.creates_cycle(pred_graph, tuple(edge)):
                pred_graph.add_edges_from([edge])
                preds.append(edge)
        else:
            remaining_proposals.append(edge)

    # Add remaining viable edges
    for edge in remaining_proposals:
        if not gutils.creates_cycle(pred_graph, tuple(edge)):
            pred_graph.add_edges_from([edge])
            preds.append(edge)
    return preds


def fuse_branches(neurograph, edges):
    cnt = 0
    for edge in edges:
        i, j = tuple(edge)
        del neurograph.proposals[edge]
        if neurograph.is_leaf(i) and neurograph.is_leaf(j):
            cnt += 1
            neurograph = simple_fusion(neurograph, edge)
        else:
            neurograph = simple_fusion(neurograph, edge)
    return neurograph


def simple_fusion(neurograph, edge):
    # Attributes
    i, j = tuple(edge)
    radii = np.array(
        [neurograph.nodes[i]["radius"], neurograph.nodes[j]["radius"]]
    )
    xyz = np.vstack([neurograph.nodes[i]["xyz"], neurograph.nodes[j]["xyz"]])

    # Add
    neurograph.add_edge(
        i, j, radius=radii, swc_id=neurograph.nodes[i]["swc_id"], xyz=xyz
    )
    return neurograph


def sample_neighbor(graph, i):
    return sample(list(graph.neighbors(i)), 1)[0]


def fusion(neurograph, edge):
    # Get nodes
    i, j = tuple(edge)
    nb_i = sample_neighbor(neurograph, i)
    nb_j = sample_neighbor(neurograph, j)

    # Get branch attributes
    branch_xyz_i = neurograph.get_branches(i, key="xyz")[0]
    branch_xyz_j = neurograph.get_branches(j, key="xyz")[0]
    branch_radii_i = neurograph.get_branches(i, key="radius")[0]
    branch_radii_j = neurograph.get_branches(j, key="radius")[0]

    # Fuse attributes
    radii = np.concatenate((np.flip(branch_radii_j), branch_radii_i))
    xyz = np.vstack((np.flip(branch_xyz_j, axis=0), branch_xyz_i))

    # Delete
    neurograph.remove_edge(i, nb_i)
    neurograph.remove_edge(j, nb_j)

    # Add
    neurograph.add_edge(
        nb_i, nb_j, radius=radii, swc_id=neurograph.nodes[i]["swc_id"], xyz=xyz
    )
    return neurograph


def save_prediction(neurograph, proposal_preds, output_dir):
    # Initializations
    corrections_dir = os.path.join(output_dir, "corrections")
    utils.mkdir(output_dir, delete=True)
    utils.mkdir(corrections_dir, delete=True)

    # Write Result
    neurograph.to_swc(output_dir)
    save_corrections(neurograph, proposal_preds, corrections_dir)


def save_corrections(neurograph, proposal_preds, output_dir):
    for cnt, (i, j) in enumerate(proposal_preds):
        # Info
        color = f"1.0 1.0 1.0"
        filename = f"merge-{cnt + 1}.swc"
        path = os.path.join(output_dir, filename)

        # Save
        xyz_i = neurograph.nodes[i]["xyz"]
        xyz_j = neurograph.nodes[j]["xyz"]
        swc_utils.save_edge(path, xyz_i, xyz_j, color=color, radius=3)
