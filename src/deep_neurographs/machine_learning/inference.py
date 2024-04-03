"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference on models that classify edge proposals.

"""

from copy import deepcopy
from random import sample

import fastremap
import networkx as nx
import numpy as np
import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from deep_neurographs import graph_utils as gutils
from deep_neurographs import reconstruction as build
from deep_neurographs import utils
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.machine_learning import feature_extraction as extracter
from deep_neurographs.machine_learning import ml_utils

BATCH_SIZE_PROPOSALS = 2000
CHUNK_SHAPE = (256, 256, 256)


def run(
    neurograph,
    model_type,
    model_path,
    img_path,
    labels_path,
    proposals,
    batch_size_proposals=BATCH_SIZE_PROPOSALS,
    confidence_threshold=0.7,
    seeds=None,
):
    if seeds:
        run_with_seeds(
            neurograph,
            model_path,
            model_type,
            img_path,
            labels_path,
            proposals,
            batch_size_proposals=batch_size_proposals,
            confidence_threshold=confidence_threshold,
            seeds=seeds,
        )
    else:
        run_without_seeds(
            neurograph,
            model_path,
            model_type,
            img_path,
            labels_path,
            proposals,
            batch_size_proposals=batch_size_proposals,
            confidence_threshold=confidence_threshold,
        )


def run_with_seeds(
    neurograph,
    model_path,
    model_type,
    img_path,
    labels_path,
    proposals,
    seeds,
    batch_size_proposals=BATCH_SIZE_PROPOSALS,
    confidence_threshold=0.7,
    output_dir=None,
):
    # build seed graphs
    # --> grow graph to some limit
    # --> run inference on each seed graph in parellel
    # --> sequentially combine results
    # --> repeat
    # check whether any proposals remain
    # --> call run_without_seeds
    pass


def run_without_seeds(
    neurograph,
    model_path,
    model_type,
    img_path,
    labels_path,
    proposals,
    batch_size_proposals=BATCH_SIZE_PROPOSALS,
    confidence_threshold=0.7,
):
    dists = [neurograph.proposal_length(edge) for edge in proposals]
    batches = utils.get_batch(np.argsort(dists), batch_size_proposals)
    model = ml_utils.load_model(model_type, model_path)
    preds = []
    for i, batch in enumerate(batches):
        # Prediction
        proposals_i = [proposals[j] for j in batch]
        preds_i = predict(
            neurograph,
            img_path,
            labels_path,
            proposals_i,
            model,
            model_type,
            confidence_threshold=confidence_threshold,
        )
        preds.extend(preds_i)

        # Merge proposals
    return preds


def predict(
    neurograph,
    img_path,
    labels_path,
    proposals,
    model,
    model_type,
    confidence_threshold=0.7,
):
    # Generate features
    features = extracter.generate_features(
        neurograph,
        model_type,
        img_path=img_path,
        labels_path=labels_path,
        proposals=proposals,
    )
    dataset = ml_utils.init_dataset(neurograph, features, model_type)

    # Run model
    proposal_probs = run_model(dataset, model, model_type)
    proposal_preds = build.get_reconstruction(
        neurograph,
        proposal_probs,
        dataset["idx_to_edge"],
        high_threshold=0.95,
        low_threshold=confidence_threshold,
    )
    return proposal_preds


def build_from_soma(
    neurograph, labels_path, chunk_origin, chunk_shape=CHUNK_SHAPE, n_hops=1
):
    swc_ids = get_swc_ids(labels_path, chunk_origin, chunk_shape)
    seed_neurograph = build_seed_neurograph(neurograph, swc_ids)
    return seed_neurograph

def get_swc_ids(path, xyz, chunk_shape, from_center=True):
    img = utils.open_tensorstore(path, "neuroglancer_precomputed")
    img = utils.read_tensorstore(
        img, xyz, chunk_shape, from_center=from_center
    )
    return set(fastremap.unique(img).astype(int))


def build_seed_neurograph(neurograph, swc_ids):
    seed_neurograph = NeuroGraph()
    for nodes in nx.connected_components(neurograph):
        i = sample_singleton(nodes)
        swc_id = int(neurograph.nodes[i]["swc_id"])
        if swc_id in swc_ids:
            seed_neurograph.update(neurograph.subgraph(nodes))
            seed_neurograph.add_swc_id(swc_id)
    return seed_neurograph


def sample_singleton(my_container):
    return sample(my_container, 1)[0]


def build_from_boundary(neurograph, pred_neurograph, boundary_components):
    new_bdd = list()
    for c in boundary_components:
        pred_neurograph, bdd = expand_boundary(neurograph, pred_neurograph, c)
        new_bdd.extend(bdd)
    return pred_neurograph, new_bdd


def expand_boundary(neurograph, pred_neurograph, component):
    bdd = list()
    for i in component:
        for j in neurograph.nodes[i]["proposals"]:
            # Extract proposals info
            edge = frozenset({i, j})
            swc_id = int(neurograph.nodes[j]["swc_id"])

            # Add component to graph
            if swc_id not in pred_neurograph.swc_ids:
                c = gutils.get_component(neurograph, j)
                pred_neurograph.add_swc_id(swc_id)
                pred_neurograph = ingest_subgraph(
                    neurograph, pred_neurograph, c
                )
                bdd.append(c)

            # Add proposal to graph
            pred_neurograph.proposals[edge] = deepcopy(
                neurograph.proposals[edge]
            )
            pred_neurograph.nodes[i]["proposals"].add(j)
            pred_neurograph.nodes[j]["proposals"].add(i)

    return pred_neurograph, bdd


def ingest_subgraph(neurograph_1, neurograph_2, node_subset):
    # Add structural components
    subgraph = nx.subgraph(neurograph_1, node_subset)
    neurograph_2.add_nodes_from(node_subset)
    neurograph_2.add_edges_from(subgraph.edges)

    # Add attributes
    for node in node_subset:
        neurograph_2.nodes[node].update(neurograph_1.nodes[node])
        neurograph_2.nodes[node]["proposals"] = set()

    for edge in subgraph.edges:
        neurograph_2.edges[edge].update(neurograph_1.edges[edge])

    return neurograph_2


def run_model(dataset, model, model_type):
    data = dataset["dataset"]
    if "Net" in model_type:
        model.eval()
        hat_y = []
        for batch in DataLoader(data, batch_size=32, shuffle=False):
            # Run model
            with torch.no_grad():
                x_i = batch["inputs"]
                hat_y_i = sigmoid(model(x_i))

            # Postprocess
            hat_y_i = np.array(hat_y_i)
            hat_y.extend(hat_y_i.tolist())
    else:
        hat_y = model.predict_proba(data["inputs"])[:, 1]
    return np.array(hat_y)
