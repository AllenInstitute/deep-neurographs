"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference with a model that classifies edge proposals.

"""

from time import time

import numpy as np
import torch
from time import time
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from deep_neurographs import graph_utils as gutils
from deep_neurographs import img_utils
from deep_neurographs import reconstruction as build
from deep_neurographs import utils
from deep_neurographs.machine_learning import (
    feature_generation,
    gnn_utils,
    ml_utils,
    seeded_inference,
)
from deep_neurographs.machine_learning.gnn_utils import toCPU

BATCH_SIZE = 1000
CONFIDENCE_THRESHOLD = 0.7


def run(
    neurograph,
    img_path,
    labels_path,
    model_type,
    model_path,
    proposals,
    search_radius,
    batch_size=BATCH_SIZE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    """
    Wrapper routine that calls the appropriate inference subroutine based on
    whether "neurograph" has nonempty "neurograph.soma_ids".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that inference will be performed on.
    img_path : str
        Path to image stored in a GCS bucket.
    labels_path : str
        Path to a segmentation mask stored in a GCS bucket.
    model_type : str
        Type of machine learning model used to perform inference.
    model_path : str
        Path to model parameters.
    proposals : list
        Proposals to be classified as accept or reject.
    search_radius : float
        Search radius used to generate proposals.
    batch_size : int, optional
        Number of proposals to generate features and classify per batch. The
        default is the global varaible "BATCH_SIZE".
    confidence_threshold : float, optional
        Threshold on acceptance probability for proposals. The default is the
        global variable "CONFIDENCE_THRESHOLD".

    Returns
    -------
    NeuroGraph
        Updated graph with accepted proposals added as edges.
    list
        Accepted proposals.

    """
    # Initializations
    assert not gutils.cycle_exists(neurograph), "Graph contains cycle!"
    model = ml_utils.load_model(model_type, model_path)

    # Open images
    driver = "n5" if ".n5" in img_path else "zarr"
    img = img_utils.open_tensorstore(img_path, driver)

    driver = "neuroglancer_precomputed"
    labels = img_utils.open_tensorstore(labels_path, driver)

    # Call inference subroutine
    if len(neurograph.soma_ids) > 0:
        neurograph, accepts = seeded_inference.run(
            neurograph,
            img,
            labels,
            model,
            model_type,
            proposals,
            search_radius,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
        )
    else:
        neurograph, accepts = run_without_seeds(
            neurograph,
            img,
            labels,
            model,
            model_type,
            proposals,
            search_radius,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold,
        )

    # Report Results
    print("\n# proposals added:", utils.reformat_number(len(accepts)))
    print("% proposals added:", round(len(accepts) / len(proposals), 4))
    return neurograph, accepts


def run_without_seeds(
    neurograph,
    img,
    labels,
    model,
    model_type,
    proposals,
    search_radius,
    batch_size=BATCH_SIZE,
    confidence_threshold=0.7,
):
    """
    Runs inference without using seeds, where batches of proposals are chosen
    with respect to their length.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that inference will be performed on.
    img : tensorstore.TensorStore
        Image stored in a GCS bucket.
    labels : tensorstore.TensorStore
        Segmentation mask stored in a GCS bucket.
    model : ..
        Machine learning model used to perform inference.
    model_type : str
        Type of machine learning model used to perform inference.
    proposals : list
        Proposals to be classified as accept or reject.
    search_radius : float
        Search radius used to generate proposals.
    batch_size : int, optional
        Number of proposals to generate features and classify per batch. The
        default is the global varaible "BATCH_SIZE".
    confidence_threshold : float, optional
        Threshold on acceptance probability for proposals. The default is the
        global variable "CONFIDENCE_THRESHOLD".

    Returns
    -------
    NeuroGraph
        Updated graph with accepted proposals added as edges.
    list
        Accepted proposals.

    """
    # Initializations
    T0 = time()
    accepts = []
    graph = neurograph.copy_graph()
    n_batches = len(proposals) // batch_size
    if "Graph" in model_type:
        batches = gnn_utils.get_batches(graph.copy(), proposals)
    else:
        dists = np.argsort([neurograph.proposal_length(p) for p in proposals])
        batches = ml_utils.get_batches(dists, batch_size)
    print("\n Initializations:", time() - T0)

    # Main
    cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = max(n_batches * 0.02, 1)
    for i, batch in enumerate(batches):
        # Init proposals
        T0 = time()
        print("batch", i)
        if "Graph" not in model_type:
            batch = [proposals[j] for j in batch]
        print("init_proposals:", time() - T0)

        # Predict
        accepts_i, graph = predict(
            neurograph,
            graph,
            img,
            labels,
            model,
            model_type,
            batch,
            search_radius,
            confidence_threshold=confidence_threshold,
        )

        # Merge proposals
        T0 = time()
        neurograph = build.fuse_branches(neurograph, accepts_i)
        accepts.extend(accepts_i)
        print("merge_proposals:", time() - T0)
        print("")

        # Report progress
        if i >= cnt * chunk_size:
            cnt, t1 = utils.report_progress(
                i + 1, n_batches, chunk_size, cnt, t0, t1
            )
    return neurograph, accepts


def predict(
    neurograph,
    graph,
    img,
    labels,
    model,
    model_type,
    batch,
    search_radius,
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    """
    Generates features and runs model on a batch of proposals.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that inference will be performed on.
    graph : networkx.Graph
        Copy of "neurograph" that does not contain attributes.
    img : tensorstore.TensorStore
        Image stored in a GCS bucket.
    labels : tensorstore.TensorStore
        Segmentation mask stored in a GCS bucket.
    model : ..
        Machine learning model used to perform inference.
    model_type : str
        Type of machine learning model used to perform inference.
    batch : dict or list
        If type(batch) is list, proposals to be classified as accept or
        reject. If type(batch) is dict, proposals to be classified as accept
        and computation graph to be used by gnn during inference.
    search_radius : float
        Search radius used to generate proposals.
    confidence_threshold : float, optional
        Threshold on acceptance probability for proposals. The default is the
        global variable "CONFIDENCE_THRESHOLD".

    Returns
    -------
    list
        Accepted proposals
    networkx.Graph
            
    """
    # Generate features
    T0 = time()
    features = feature_generation.run(
        neurograph,
        model_type,
        batch,
        search_radius,
        img,
        labels=labels,
    )
    computation_graph = batch["graph"] if type(batch) is dict else None
    dataset = ml_utils.init_dataset(
        neurograph, features, model_type, computation_graph=computation_graph
    )
    print("feature_generation:", time() - T0)

    # Run model
    T0 = time()
    idx_to_edge = get_idxs(dataset, model_type)
    proposal_probs = run_model(dataset, model, model_type)
    print("run_model:", time() - T0)

    T0 = time()
    accepts, graph = build.get_accepted_proposals(
        neurograph,
        graph,
        proposal_probs,
        idx_to_edge,
        high_threshold=0.9,
        threshold=confidence_threshold,
    )
    print("get_accepted_proposals:", time() - T0)
    
    return accepts, graph


# --- run machine learning model ---
def run_model(dataset, model, model_type):
    if "Graph" in model_type:
        return run_gnn_model(dataset.data, model, model_type)
    elif "Net" in model_type:
        return run_nn_model(dataset, model)
    else:
        hat_y = model.predict_proba(dataset["dataset"]["inputs"])
        return np.array(hat_y[:, 1])


def run_nn_model(dataset, model):
    hat_y = []
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset["dataset"], batch_size=32):
            # Run model
            x_i = batch["inputs"]
            hat_y_i = sigmoid(model(x_i))

            # Postprocess
            hat_y_i = np.array(hat_y_i)
            hat_y.extend(hat_y_i[:, 0].tolist())
    return np.array(hat_y)


def run_gnn_model(data, model, model_type):
    model.eval()
    with torch.no_grad():
        if "Hetero":
            x_dict, edge_index_dict, edge_attr_dict = gnn_utils.get_inputs(
                data, model_type
            )
            hat_y = sigmoid(model(x_dict, edge_index_dict, edge_attr_dict))
            idx = len(data["proposal"]["y"])
        else:
            x, edge_index = gnn_utils.get_inputs(data, model_type)
            hat_y = sigmoid(model(x, edge_index))
            idx = len(data.proposals)
    return toCPU(hat_y[0:idx, 0])


# --- utils ---
def get_idxs(dataset, model_type):
    """
    Gets dictionary from "dataset" that maps indices (from feature matrix) to
    proposal ids.

    Parameters
    ----------
    dataset : ProposalDataset
        Dataset that contains features generated from proposals.
    model_type : str
        Type of model used to perform inference.

    Returns
    -------
    dict
        Dictionary that maps indices (from feature matrix) to proposal ids.

    """
    if "Graph" in model_type:
        return dataset.idxs_proposals["idx_to_edge"]
    else:
        return dataset["idx_to_edge"]
