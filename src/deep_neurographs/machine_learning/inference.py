"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference on models that classify edge proposals.

"""

import numpy as np
import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

CHUNK_SHAPE = (512, 512, 512)


from copy import deepcopy
from random import sample

import fastremap
import networkx as nx

from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.neurograph import NeuroGraph




def run(
    neurograph,
    model_type,
    model_path,
    labels_path,
    proposals,
    batch_size_proposals=2000,
    confidence_threshold=0.7,
    output_dir=None,
    seeds=None,
):
    if seeds:
        run_with_seeds(
            neurograph,
            model_type,
            model_path,
            labels_path,
            proposals,
            confidence_threshold=0.7,
            output_dir=None,
            seeds=None,
        )
    else:
        run_without_seeds(
            neurograph,
            model_type,
            model_path,
            labels_path,
            proposals,
            confidence_threshold=0.7,
            output_dir=None,
        )
    


def run_with_seeds(
    neurograph,
    model_type,
    model_path,
    labels_path,
    proposals,
    seeds,
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
    model_type,
    model_path,
    labels_path,
    proposals,
    confidence_threshold=0.7,
    output_dir=None,
):
    # form batches of proposals wrt distance
    # --> generate features
    # --> run prediction
    # --> parse predictions and merge proposals
    pass


def build_from_soma(
    neurograph, labels_path, chunk_origin, chunk_shape=CHUNK_SHAPE, n_hops=1
):
    swc_ids = get_swc_ids(labels_path, chunk_origin, chunk_shape)
    seed_neurograph = build_seed_neurograph(neurograph, swc_ids)


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


def predict(dataset, model, model_type):
    accuracy = []
    accuracy_baseline = []
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
            y_i = np.array(batch["targets"])
            hat_y.extend(hat_y_i.tolist())
            accuracy_baseline.extend((y_i > 0).tolist())
            accuracy.extend(((hat_y_i > 0.5) == (y_i > 0)).tolist())
        accuracy = np.mean(accuracy)
        accuracy_baseline = np.sum(accuracy_baseline) / len(accuracy_baseline)
        print("Accuracy +/-:", accuracy - accuracy_baseline)
    else:
        hat_y = model.predict_proba(data["inputs"])[:, 1]
    return np.array(hat_y)
