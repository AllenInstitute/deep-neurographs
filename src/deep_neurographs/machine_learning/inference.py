"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference on models that classify edge proposals.

"""

from copy import deepcopy
from time import time

import fastremap
import networkx as nx
import numpy as np
import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from deep_neurographs import graph_utils as gutils
from deep_neurographs import reconstruction as build
from deep_neurographs import utils
from deep_neurographs.machine_learning import feature_generation, ml_utils
from deep_neurographs.neurograph import NeuroGraph

BATCH_SIZE_PROPOSALS = 1000
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
            seeds,
            batch_size_proposals=batch_size_proposals,
            confidence_threshold=confidence_threshold,
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
    progress_bar=True,
):
    # Initializations
    graph = neurograph.copy_graph()
    dists = [neurograph.proposal_length(edge) for edge in proposals]
    batches = utils.get_batches(np.argsort(dists), batch_size_proposals)
    model = ml_utils.load_model(model_type, model_path)
    n_batches = 1 + len(proposals) // BATCH_SIZE_PROPOSALS
    assert not gutils.cycle_exists(graph), "Google prediction contains cycle"

    # Run
    accepts = []
    progress_cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = max(int(n_batches * 0.02), 1)
    for i, batch in enumerate(batches):
        # Prediction
        proposals_i = [proposals[j] for j in batch]
        accepts_i = predict(
            neurograph,
            graph,
            img_path,
            labels_path,
            proposals_i,
            model,
            model_type,
            confidence_threshold=confidence_threshold,
        )

        # Merge proposals
        neurograph = build.fuse_branches(neurograph, accepts_i)
        accepts.extend(accepts_i)

        # Report progress
        if i > progress_cnt * chunk_size and progress_bar:
            progress_cnt, t1 = report_progress(
                i, n_batches, chunk_size, progress_cnt, t0, t1
            )
            t0, t1 = utils.init_timers()

    return neurograph, accepts


def predict(
    neurograph,
    graph,
    img_path,
    labels_path,
    proposals,
    model,
    model_type,
    confidence_threshold=0.7,
):
    # Generate features
    features = feature_generation.run(
        neurograph,
        model_type,
        img_path,
        labels_path=labels_path,
        proposals=proposals,
    )
    dataset = ml_utils.init_dataset(neurograph, features, model_type)

    # Run model
    proposal_probs = run_model(dataset, model, model_type)
    accepts = build.get_accepted_proposals(
        neurograph,
        graph,
        proposal_probs,
        dataset["idx_to_edge"],
        high_threshold=0.95,
        low_threshold=confidence_threshold,
    )
    return accepts


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
        i = utils.sample_singleton(nodes)
        swc_id = int(neurograph.nodes[i]["swc_id"])
        if swc_id in swc_ids:
            seed_neurograph.update(neurograph.subgraph(nodes))
            seed_neurograph.add_swc_id(swc_id)
    return seed_neurograph


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


# Utils
def report_progress(current, total, chunk_size, cnt, t0, t1):
    eta = get_eta(current, total, chunk_size, t1)
    runtime = get_runtime(current, total, chunk_size, t0, t1)
    utils.progress_bar(current, total, eta=eta, runtime=runtime)
    return cnt + 1, time()


def get_eta(current, total, chunk_size, t0, return_str=True):
    chunk_runtime = time() - t0
    remaining = total - current
    eta = remaining * (chunk_runtime / chunk_size)
    t, unit = utils.time_writer(eta)
    return f"{round(t, 4)} {unit}" if return_str else eta


def get_runtime(current, total, chunk_size, t0, t1):
    eta = get_eta(current, total, chunk_size, t1, return_str=False)
    total_runtime = time() - t0 + eta
    t, unit = utils.time_writer(total_runtime)
    return f"{round(t, 4)} {unit}"
