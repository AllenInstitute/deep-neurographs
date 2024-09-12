"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference with a model that classifies edge proposals.

"""

from time import time

import networkx as nx
import numpy as np
import torch
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

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

BATCH_SIZE = 1600
CONFIDENCE_THRESHOLD = 0.7


class InferenceEngine:
    """
    Class that runs inference with a machine learning model that has been
    trained to classify edge proposals.

    """

    def __init__(
        self,
        img_path,
        model_path,
        model_type,
        search_radius,
        batch_size=BATCH_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        downsample_factor=0,
    ):
        """
        Initializes an inference engine by loading images and setting class
        attributes.

        Parameters
        ----------
        img_path : str
            Path to image stored in a GCS bucket.
        model_path : str
            Path to model parameters.
        model_type : str
            Type of machine learning model used to perform inference.
        search_radius : float
            Search radius used to generate proposals.
        batch_size : int, optional
            Number of proposals to generate features and classify per batch. The
            default is the global varaible "BATCH_SIZE".
        confidence_threshold : float, optional
            Threshold on acceptance probability for proposals. The default is the
            global variable "CONFIDENCE_THRESHOLD".
        downsample_factor : int, optional
            Downsampling factor that accounts for which level in the image pyramid
            the voxel coordinates must index into. The default is 0.

        Returns
        -------
        None

        """
        # Set class attributes
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.is_gnn = True if "Graph" in model_type else False
        self.model_type = model_type
        self.search_radius = search_radius
        self.threshold = confidence_threshold

        # Load image and model
        driver = "n5" if ".n5" in img_path else "zarr"
        self.img = img_utils.open(img_path, driver)
        self.model = ml_utils.load_model(model_type, model_path)

    def run(self, neurograph, proposals):
        """
        Runs inference by forming batches of proposals, then performing the
        following steps for each batch: (1) generate features, (2) classify
        proposals by running model, and (3) adding each accepted proposal as
        an edge to "neurograph" if it does not create a cycle.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that inference will be performed on.
        proposals : list
            Proposals to be classified as accept or reject.

        Returns
        -------
        NeuroGraph
            Updated graph with accepted proposals added as edges.
        list
            Accepted proposals.

        """
        assert not gutils.cycle_exists(neurograph), "Graph contains cycle!"
        accepts = []
        batches = self.get_batches(neurograph, proposals)
        for batch in tqdm(batches, desc="Inference"):
            # Predict
            dataset = self.get_batch_dataset(neurograph, batch)
            preds = self.run_model(dataset)

            # Update graph
            batch_accepts = get_accepted_proposals(neurograph, preds)
            for proposal in batch_accepts:
                neurograph.merge_proposal(frozenset(proposal))
            accepts.extend(batch_accepts)

        # Report Results
        print("\n# proposals added:", utils.reformat_number(len(accepts)))
        print("% proposals added:", round(len(accepts) / len(proposals), 4))
        return neurograph, accepts

    def get_batches(self, neurograph, proposals):
        """
        Generates batches of proposals.

        Parameters
        ----------
        proposals : list
            Proposals for which batches are to be generated from.

        Returns
        -------
        list
            List of batches where each batch is a subset of "proposals".

        """
        if self.is_gnn:
            return gnn_utils.get_batches(neurograph.copy(), proposals)
        else:
            dists = np.argsort(
                [neurograph.proposal_length(p) for p in proposals]
            )
            batches = list()
            for idxs in ml_utils.get_batches(dists, self.batch_size):
                batches.append([proposals[i] for i in idxs])
            return batches

    def get_batch_dataset(self, neurograph, batch):
        """
        Generates features and initializes dataset that can be input to a
        machine learning model.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that inference will be performed on.
        batch : list
            Proposals to be classified.

        Returns
        -------
        ...

        """
        # Generate features
        features = feature_generation.run(
            neurograph,
            self.img,
            self.model_type,
            batch,
            self.search_radius,
            downsample_factor=self.downsample_factor,
        )

        # Initialize dataset
        computation_graph = batch["graph"] if type(batch) is dict else None
        dataset = ml_utils.init_dataset(
            neurograph,
            features,
            self.model_type,
            computation_graph=computation_graph,
        )
        return dataset

    def run_model(self, dataset):
        """
        Runs the model on the given dataset to generate and filter
        predictions.

        Parameters
        ----------
        dataset : ...
            Dataset on which the model inference is to be run.

        Returns
        -------
        dict
            A dictionary that maps a proposal to the model's prediction (i.e.
            probability). Note that this dictionary only contains proposals
            whose predicted probability is greater the 
        """
        # Get predictions
        if self.is_gnn:
            preds = run_gnn_model(dataset.data, self.model, self.model_type)
        elif "Net" in self.model_type:
            preds = run_nn_model(dataset, self.model)
        else:
            data = dataset["dataset"]["inputs"]
            preds = np.array(self.model.predict_proba(data)[:, 1])

        # Filter preds
        idxs = get_idxs(dataset, self.model_type)
        return {idxs[i]: p for i, p in enumerate(preds) if p > self.threshold}


# --- run machine learning model ---
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
        if "Hetero" in model_type:
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


# --- Accepting proposals ---
def get_accepted_proposals(neurograph, preds, high_threshold=0.9):
    """
    Determines which proposals to accept based on prediction scores and the
    specified threshold.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that proposals belong to.
    preds : dict
        Dictionary that maps proposal ids to probability generated from
        machine learning model.
    high_threshold : float, optional
        Threshold value for separating the best proposals from the rest. The
        default is 0.9.

    Returns
    -------
    list
        Proposals to be added as edges to "neurograph".

    """
    # Partition proposals into best and the rest
    best_proposals, proposals = separate_best(
        preds, neurograph.simple_proposals(), high_threshold
    )

    # Determine which proposals to accept
    accepts = list()
    accepts.extend(filter_proposals(neurograph, best_proposals))
    accepts.extend(filter_proposals(neurograph, proposals))
    return accepts


def separate_best(preds, simple_proposals, high_threshold):
    """
    Splits "preds" into two separate dictionaries such that one contains the
    best proposals (i.e. simple proposals with high confidence) and the other
    contains all other proposals.

    Parameters
    ----------
    preds : dict
        Dictionary that maps proposal ids to probability generated from
        machine learning model.
    simple_proposals : list
        List of simple proposals.
    high_threshold : float
        Threshold on acceptance probability for proposals.

    Returns
    -------
    list
        Proposal IDs determined to be the best.
    list
        All other proposal IDs.

    """
    best_probs, probs = list(), list()
    best_proposals, proposals = list(), list()
    for proposal, prob in preds.items():
        if proposal in simple_proposals and prob > high_threshold:
            best_proposals.append(proposal)
            best_probs.append(prob)
        else:
            proposals.append(proposal)
            probs.append(prob)
    best_idxs = np.argsort(best_probs)
    idxs = np.argsort(probs)
    return np.array(best_proposals)[best_idxs], np.array(proposals)[idxs]


def filter_proposals(graph, proposals):
    """
    Filters a list of proposals by removing the ones that create a cycle when
    added to "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to which edges are being added.
    proposals : list[frozenset]
        List of proposals to be checked.

    Returns
    -------
    list[frozenset]
        List of proposals that do not create a cycle when iteratively added to
        "graph".

    """
    accepts = list()
    for i, j in proposals:
        nodes_i = set(gutils.get_component(graph, i))
        nodes_j = set(gutils.get_component(graph, j))
        if nodes_i.isdisjoint(nodes_j):
            subgraph_i = graph.subgraph(nodes_i)
            subgraph_j = graph.subgraph(nodes_j)
            subgraph = nx.union(subgraph_i, subgraph_j)
            created_cycle, _ = gutils.creates_cycle(subgraph, (i, j))
            if not created_cycle:
                graph.add_edge(i, j)
                accepts.append((i, j))
    graph.remove_edges_from(accepts)
    return accepts


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
