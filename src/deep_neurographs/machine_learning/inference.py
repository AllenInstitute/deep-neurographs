"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference with a model that classifies edge proposals.

"""

import numpy as np
import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader

from deep_neurographs import graph_utils as gutils
from deep_neurographs import reconstruction as build
from deep_neurographs import utils
from deep_neurographs.machine_learning import (
    feature_generation,
    gnn_utils,
    ml_utils,
)
from deep_neurographs.machine_learning.gnn_utils import toCPU

BATCH_SIZE_PROPOSALS = 1000


def run(
    neurograph,
    model_type,
    model_path,
    img_path,
    labels_path,
    proposals,
    search_radius,
    batch_size_proposals=BATCH_SIZE_PROPOSALS,
    confidence_threshold=0.7,
):
    """
    ...

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that inference will be performed on.
    model_type : str
        Type of machine learning model used to perform inference.
    model_path : str
        Path to model parameters.
    img_path : str
        Path to raw image stored in a GCS bucket.
    labels_path : str
        Path to a segmentation mask stored in a GCS bucket.
    proposals : dict
        Proposals to be classified as accept or reject.
    search_radius : float
        Search radius used to generate proposals.
    batch_size_proposals : int
        Number of proposals to generate features for and classify at a given
        time
    confidence_threshold : float
        Threshold on acceptance probability for proposals.

    Returns
    -------
    NeuroGraph
        Updated graph with accepted proposals added as edges.
    list
        Accepted proposals.

    """
    # Initializations
    assert not gutils.cycle_exists(neurograph), "NeuroGraph contains cycle!"
    graph = neurograph.copy_graph()
    dists = [neurograph.proposal_length(edge) for edge in proposals]
    batches = utils.get_batches(np.argsort(dists), batch_size_proposals)
    model = ml_utils.load_model(model_type, model_path)
    n_batches = 1 + len(proposals) // BATCH_SIZE_PROPOSALS

    # Run
    accepts = []
    progress_cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = max(int(n_batches * 0.02), 1)
    for i, batch in enumerate(batches):        
        # Prediction
        proposals_i = [proposals[j] for j in batch]
        accepts_i, graph = predict(
            neurograph,
            graph,
            img_path,
            labels_path,
            proposals_i,
            model,
            model_type,
            search_radius,
            confidence_threshold=confidence_threshold,
        )

        # Merge proposals
        neurograph = build.fuse_branches(neurograph, accepts_i)
        accepts.extend(accepts_i)

        # Report progress
        if i >= progress_cnt * chunk_size:
            progress_cnt, t1 = utils.report_progress(
                i + 1, n_batches, chunk_size, progress_cnt, t0, t1
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
    search_radius,
    confidence_threshold=0.7,
):
    # Generate features
    features = feature_generation.run(
        neurograph,
        model_type,
        search_radius,
        img_path,
        labels_path=labels_path,
        proposals=proposals,
    )
    dataset = ml_utils.init_dataset(neurograph, features, model_type)

    # Run model
    idx_to_edge = get_idxs(dataset, model_type)
    proposal_probs = run_model(dataset, model, model_type)
    accepts, graph = build.get_accepted_proposals(
        neurograph,
        graph,
        proposal_probs,
        idx_to_edge,
        search_radius,
        high_threshold=0.9,
        threshold=confidence_threshold,
    )
    return accepts, graph


def run_model(dataset, model, model_type):
    if "Graph" in model_type:
        return run_graph_model(dataset.data, model, model_type)
    elif "Net" in model_type:
        model.eval()
        hat_y = []
        data = dataset["dataset"]
        for batch in DataLoader(data, batch_size=32, shuffle=False):
            # Run model
            with torch.no_grad():
                x_i = batch["inputs"]
                hat_y_i = sigmoid(model(x_i))

            # Postprocess
            hat_y_i = np.array(hat_y_i)
            hat_y.extend(hat_y_i[:, 0].tolist())
    else:
        data = dataset["dataset"]
        hat_y = model.predict_proba(data["inputs"])[:, 1]
    return np.array(hat_y)


def run_graph_model(data, model, model_type):
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
    if "Graph" in model_type:
        return dataset.idxs_proposals["idx_to_edge"]
    else:
        return dataset["idx_to_edge"]
