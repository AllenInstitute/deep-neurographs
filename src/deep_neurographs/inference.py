"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that executes the full GraphTrace inference pipeline.

    Inference Algorithm:
        1. Graph Construction
            Builds a graph from neuron fragments.

        2. Proposal Generation
            Generates proposals for potential connections between fragments.

        3. Proposal Classification
            a. Feature Generation
                Extracts features from proposals and graph for a machine
                learning model.
            b. Predict with Graph Neural Network (GNN)
                Runs a GNN to classify proposals as accept/reject
                based on the learned features.
            c. Merge Accepted Proposals
                Adds accepted proposals to the fragments graph as edges.

Note: Steps 2 and 3 of the inference pipeline can be iterated in a loop that
      repeats multiple times by calling the routine "run_schedule" within the
      InferencePipeline class.

"""

import os
from datetime import datetime
from time import time

import networkx as nx
import numpy as np
import torch
from torch.nn.functional import sigmoid
from tqdm import tqdm

from deep_neurographs import fragment_filtering
from deep_neurographs.fragments_graph import FragmentsGraph
from deep_neurographs.machine_learning import heterograph_datasets
from deep_neurographs.machine_learning.feature_generation import (
    FeatureGenerator,
)
from deep_neurographs.utils import ml_util, util


class InferencePipeline:
    """
    Class that executes the full GraphTrace inference pipeline by performing
    the following steps: (1) Graph Construction, (2) Proposal Generation, and
    (3) Proposal Classification.

    """

    def __init__(
        self,
        brain_id,
        segmentation_id,
        img_path,
        model_path,
        output_dir,
        config,
        device="cpu",
        is_multimodal=False,
        segmentation_path=None,
        somas_path=None,
        s3_dict=None,
    ):
        """
        Initializes an object that executes the full GraphTrace inference
        pipeline.

        Parameters
        ----------
        brain_id : str
            Identifier for the whole-brain dataset.
        segmentation_id : str
            Identifier for the segmentation model that generated fragments.
        img_path : str
            Path to the whole-brain image stored in a GCS or S3 bucket.
        model_path : str
            Path to machine learning model parameters.
        output_dir : str
            Directory where the results of the inference will be saved.
        config : Config
            Configuration object containing parameters and settings required
            for the inference pipeline.
        device : str, optional
            ...
        is_multimodal : bool, optional
            ...
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. The default is None.
        somas_path : str, optional
            Path to a txt file containing xyz coordinates of detected somas.
            The default is None.
        s3_dict : dict, optional
            ...

        Returns
        -------
        None

        """
        # Instance attributes
        self.accepted_proposals = list()
        self.model_path = model_path
        self.brain_id = brain_id
        self.segmentation_id = segmentation_id
        self.segmentation_path = segmentation_path
        self.somas_path = somas_path
        self.s3_dict = s3_dict

        # Extract config settings
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

        # Inference engine
        self.inference_engine = InferenceEngine(
            img_path,
            self.model_path,
            self.graph_config.search_radius,
            accept_threshold=self.ml_config.threshold,
            anisotropy=self.ml_config.anisotropy,
            batch_size=self.ml_config.batch_size,
            device=device,
            multiscale=self.ml_config.multiscale,
            segmentation_path=segmentation_path,
            is_multimodal=is_multimodal,
        )

        # Set output directory
        self.output_dir = output_dir
        util.mkdir(self.output_dir, delete=True)

        # Initialize logger
        log_path = os.path.join(self.output_dir, "runtimes.txt")
        self.log_handle = open(log_path, 'a')

    # --- Core ---
    def run(self, fragments_pointer):
        """
        Executes the full inference pipeline.

        Parameters
        ----------
        fragments_pointer : Any
            Pointer to SWC files used to build an instance of FragmentGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        None

        """
        # Initializations
        self.log_experiment()
        self.write_metadata()
        t0 = time()

        # Main
        self.build_graph(fragments_pointer)
        self.generate_proposals()
        self.classify_proposals()
        self.save_results()

        # Finish
        t, unit = util.time_writer(time() - t0)
        self.report_graph(prefix="\nFinal")
        self.report(f"Total Runtime: {round(t, 4)} {unit}\n")

    def run_schedule(self, fragments_pointer, radius_schedule):
        # Initializations
        self.log_experiment()
        self.write_metadata()
        t0 = time()

        # Main
        self.build_graph(fragments_pointer)
        for round_id, radius in enumerate(radius_schedule):
            round_id += 1
            self.report(f"\n--- Round {round_id}:  Radius = {radius} ---")
            self.generate_proposals(radius)
            self.classify_proposals()
            self.report_graph(prefix="Current")
        self.save_results()

        # Finish
        t, unit = util.time_writer(time() - t0)
        self.report_graph(prefix="\nFinal")
        self.report(f"Total Runtime: {round(t, 4)} {unit}\n")

    def build_graph(self, fragments_pointer):
        """
        Builds a graph from the given fragments.

        Parameters
        ----------
        fragment_pointer : dict, list, str
            Pointer to SWC files used to build an instance of FragmentGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        None

        """
        self.report("Step 1: Build Fragments Graph")
        t0 = time()

        # Initialize graph
        self.graph = FragmentsGraph(
            anisotropy=self.graph_config.anisotropy,
            min_size=self.graph_config.min_size,
            node_spacing=self.graph_config.node_spacing,
            prune_depth=self.graph_config.prune_depth,
            remove_high_risk_merges=self.graph_config.remove_high_risk_merges,
            segmentation_path=self.segmentation_path,
            smooth_bool=self.graph_config.smooth_bool,
            somas_path=self.somas_path,
            verbose=True,
        )
        self.graph.load_fragments(fragments_pointer)
        self.filter_fragments()

        # Save valid labels and current graph
        swc_dir = os.path.join(self.output_dir, "swcs")
        valid_labels_path = os.path.join(self.output_dir, "valid_labels.txt")
        self.graph.to_zipped_swcs_parallelized(swc_dir, sampling_rate=4)
        self.graph.save_labels(valid_labels_path)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.report_graph(prefix="\nInitial")
        self.report(f"Module Runtime: {round(t, 4)} {unit}\n")

    def filter_fragments(self):
        self.graph = fragment_filtering.remove_curvy(self.graph, 200)
        if self.graph_config.remove_doubles:
            self.graph = fragment_filtering.remove_doubles(
                self.graph, 200, self.graph_config.node_spacing
            )

    def generate_proposals(self, radius=None):
        """
        Generates proposals for the fragments graph based on the specified
        configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initializations
        self.report("Step 2: Generate Proposals")
        if radius is None:
            radius = self.graph_config.search_radius

        # Main
        t0 = time()
        self.graph.generate_proposals(
            radius,
            complex_bool=self.graph_config.complex_bool,
            long_range_bool=self.graph_config.long_range_bool,
            proposals_per_leaf=self.graph_config.proposals_per_leaf,
            trim_endpoints_bool=self.graph_config.trim_endpoints_bool,
        )
        n_proposals = util.reformat_number(self.graph.n_proposals())

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.report(f"# Proposals: {n_proposals}")
        self.report(f"Module Runtime: {round(t, 4)} {unit}\n")

    def classify_proposals(self):
        """
        Classifies proposals by calling "self.inference_engine". This routine
        generates features and runs a GNN to make predictions. Proposals with
        a prediction above "self.threshold" are accepted and added to the
        graph as an edge.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initializations
        self.report("Step 3: Run Inference")
        proposals = self.graph.list_proposals()
        n_proposals = max(len(proposals), 1)

        # Main
        t0 = time()
        self.graph, accepts = self.inference_engine.run(self.graph, proposals)
        self.accepted_proposals.extend(accepts)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.report(f"# Accepted: {util.reformat_number(len(accepts))}")
        self.report(f"% Accepted: {round(len(accepts) / n_proposals, 4)}")
        self.report(f"Module Runtime: {round(t, 4)} {unit}\n")

    def save_results(self):
        """
        Saves the processed results from running the inference pipeline,
        namely the corrected SWC files and a list of the merged SWC ids.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Save result on local machine
        swc_dir = os.path.join(self.output_dir, "corrected-swcs")
        self.graph.to_zipped_swcs_parallelized(swc_dir, min_size=50)
        self.save_connections()
        self.write_metadata()

        # Save result on s3 (if applicable)
        if self.s3_dict is not None:
            self.save_to_s3()

    def save_to_s3(self):
        """
        Saves a corrected swc files to s3 along with metadata and runtimes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        bucket_name = self.s3_dict["bucket_name"]
        for name in os.listdir(self.output_dir):
            if os.path.isdir(os.path.join(self.output_dir, name)):
                dir_path = os.path.join(self.output_dir, name)
                prefix = os.path.join(self.s3_dict["prefix"], name)
                util.dir_to_s3(dir_path, bucket_name, prefix)
            else:
                local_path = os.path.join(self.output_dir, name)
                s3_path = os.path.join(self.s3_dict["prefix"], name)
                util.file_to_s3(local_path, bucket_name, s3_path)
        print("Results written to S3 prefix -->", self.s3_dict["prefix"])

    # --- io ---
    def save_connections(self, round_id=None):
        """
        Writes the accepted proposals from the graph to a text file. Each line
        contains the two swc ids as comma separated values.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        suffix = f"-{round_id}" if round_id else ""
        path = os.path.join(self.output_dir, f"connections{suffix}.txt")
        with open(path, "w") as f:
            for id_1, id_2 in self.graph.merged_ids:
                f.write(f"{id_1}, {id_2}" + "\n")

    def write_metadata(self):
        """
        Writes metadata about the current pipeline run to a JSON file.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        metadata = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "brain_id": self.brain_id,
            "segmentation_id": self.segmentation_id,
            "min_fragment_size": f"{self.graph_config.min_size}um",
            "node_spacing": self.graph_config.node_spacing,
            "remove_doubles": self.graph_config.remove_doubles,
            "use_somas": self.segmentation_path and self.somas_path,
            "complex_proposals": self.graph_config.complex_bool,
            "long_range_bool": self.graph_config.long_range_bool,
            "proposals_per_leaf": self.graph_config.proposals_per_leaf,
            "search_radius": f"{self.graph_config.search_radius}um",
            "model_name": os.path.basename(self.model_path),
            "accept_threshold": self.ml_config.threshold,
        }
        path = os.path.join(self.output_dir, "metadata.json")
        util.write_json(path, metadata)

    # --- Summaries ---
    def report(self, txt):
        print(txt)
        self.log_handle.write(txt)
        self.log_handle.write("\n")

    def log_experiment(self):
        self.report("\nExperiment Overview")
        self.report("-------------------------------------------------------")
        self.report(f"Brain_ID: {self.brain_id}")
        self.report(f"Segmentation_ID: {self.segmentation_id}")
        self.report("\n")

    def report_graph(self, prefix="\n"):
        """
        Prints an overview of the graph's structure and memory usage.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Compute values
        n_components = nx.number_connected_components(self.graph)
        n_components = util.reformat_number(n_components)
        n_nodes = util.reformat_number(self.graph.number_of_nodes())
        n_edges = util.reformat_number(self.graph.number_of_edges())
        usage = round(util.get_memory_usage(), 2)

        # Report
        self.report(f"{prefix} Graph")
        self.report(f"# Connected Components: {n_components}")
        self.report(f"# Nodes: {n_nodes}")
        self.report(f"# Edges: {n_edges}")
        self.report(f"Memory Consumption: {usage} GBs")


class InferenceEngine:
    """
    Class that runs inference with a machine learning model that has been
    trained to classify edge proposals.

    """

    def __init__(
        self,
        img_path,
        model_path,
        radius,
        accept_threshold=0.6,
        anisotropy=[1.0, 1.0, 1.0],
        batch_size=2000,
        device=None,
        multiscale=1,
        segmentation_path=None,
        is_multimodal=False
    ):
        """
        Initializes an inference engine by loading images and setting class
        attributes.

        Parameters
        ----------
        img_path : str
            Path to image.
        model_path : str
            Path to machine learning model weights.
        radius : float
            Search radius used to generate proposals.
        accept_threshold : float, optional
            Threshold for accepting proposals, where proposals with predicted
            likelihood above this threshold are accepted. The default is 0.6.
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        batch_size : int, optional
            Number of proposals to classify in each batch.The default is 2000.
        multiscale : int, optional
            Level in the image pyramid that voxel coordinates must index into.
            The default is 1.
        segmentation_path : str or None, optional
            ...
        is_multimodal : bool, optional
            ...

        Returns
        -------
        None

        """
        # Set class attributes
        self.batch_size = batch_size
        self.device = "cpu" if device is None else device
        self.radius = radius
        self.threshold = accept_threshold

        # Features
        self.feature_generator = FeatureGenerator(
            img_path,
            multiscale,
            anisotropy=anisotropy,
            segmentation_path=segmentation_path,
            is_multimodal=is_multimodal
        )

        # Model
        self.model = ml_util.load_model(model_path)
        if "cuda" in device:
            self.model = self.model.to(self.device)

    def run(self, fragments_graph, proposals):
        """
        Runs inference by forming batches of proposals, then performing the
        following steps for each batch: (1) generate features, (2) classify
        proposals by running model, and (3) adding each accepted proposal as
        an edge to "fragments_graph" if it does not create a cycle.

        Parameters
        ----------
        fragments_graph : FragmentsGraph
            Graph that inference will be performed on.
        proposals : list
            Proposals to be classified as accept or reject.

        Returns
        -------
        FragmentsGraph
            Updated graph with accepted proposals added as edges.
        list
            Accepted proposals.

        """
        flagged = set()  # get_large_proposal_components(fragments_graph, 4)
        proposals = set(proposals)
        with tqdm(total=len(proposals), desc="Inference") as pbar:
            accepts = list()
            while len(proposals) > 0:
                # Predict
                batch = self.get_batch(fragments_graph, proposals, flagged)
                dataset = self.get_batch_dataset(fragments_graph, batch)
                preds = self.predict(dataset)

                # Update graph
                for p in get_accepts(fragments_graph, preds, self.threshold):
                    fragments_graph.merge_proposal(p)
                    accepts.append(p)
                pbar.update(len(batch["proposals"]))
            #fragments_graph.absorb_reducibles()  # - extremely slow
        return fragments_graph, accepts

    def get_batch(self, graph, proposals, flagged_proposals):
        """
        Generates a batch of proposals.

        Parameters
        ----------
        graph : FragmentsGraph
            Graph that proposals were generated from.
        proposals : List[frozenset]
            Proposals for which batch is to be generated from.
        flagged_proposals : List[frozenset]
            List of proposals that are part of a "large" connected component
            in the proposal induced subgraph of "fragments_graph".

        Returns
        -------
        dict
            Batch which consists of a subset of "proposals" and the
            computation graph if the model type is a gnn.

        """
        batch = ml_util.get_batch(
            graph, proposals, self.batch_size, flagged_proposals
        )
        return batch

    def get_batch_dataset(self, graph, batch):
        """
        Generates features and initializes dataset that can be input to a
        machine learning model.

        Parameters
        ----------
        graph : FragmentsGraph
            Graph that inference will be performed on.
        batch : list
            Proposals to be classified.

        Returns
        -------
        ...

        """
        features = self.feature_generator.run(graph, batch, self.radius)
        return heterograph_datasets.init(graph, features,  batch["graph"])

    def predict(self, dataset):
        """
        Runs the model on the given dataset to generate and filter
        predictions.

        Parameters
        ----------
        data : HeteroGeneousDataset
            Dataset containing graph information, including feature matrices
            and other relevant attributes needed for GNN input.
        Returns
        -------
        dict
            A dictionary that maps a proposal to the model's prediction (i.e.
            probability).

        """
        preds = predict(self.model, dataset.data, self.device)
        idxs = dataset.idxs_proposals["idx_to_id"]
        return {idxs[i]: p for i, p in enumerate(preds)}


# --- Accepting Proposals ---
def predict(model, data, device=None):
    """
    Generates predictions using a Graph Neural Network (GNN) on the given
    dataset.

    Parameters:
    ----------
    model : torch.nn.Module
        GNN model used to generate predictions. It should accept node
        features, edge indices, and edge attributes as input and output
        predictions.
    data : dict
        Dataset containing graph information, including feature matrices
        and other relevant attributes needed for GNN input.
    device : str, optional
        The device (CPU or GPU) on which the prediction will be run. The
        default is None.

    Returns:
    -------
    torch.Tensor
        A tensor of predictions, converted to CPU, for the 'proposal' entries
        in the dataset. Only the relevant predictions for 'proposal' nodes are
        returned.

    """
    with torch.no_grad():
        x, edge_index, edge_attr = ml_util.get_inputs(data, device)
        preds = sigmoid(model(x, edge_index, edge_attr))
    return ml_util.toCPU(preds[0:len(data["proposal"]["y"]), 0])


def get_accepts(fragments_graph, preds, threshold, high_threshold=0.9):
    """
    Determines which proposals to accept based on prediction scores and the
    specified threshold.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
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
        Proposals to be added as edges to "fragments_graph".

    """
    # Partition proposals into best and the rest
    preds = {k: v for k, v in preds.items() if v > threshold}
    best_proposals, proposals = separate_best(
        preds, fragments_graph.simple_proposals(), high_threshold
    )

    # Determine which proposals to accept
    accepts = list()
    accepts.extend(filter_proposals(fragments_graph, best_proposals))
    accepts.extend(filter_proposals(fragments_graph, proposals))
    fragments_graph.remove_edges_from(map(tuple, accepts))
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
        if not nx.has_path(graph, i, j):
            graph.add_edge(i, j)
            accepts.append(frozenset({i, j}))
    return accepts


def sort_proposals(fragments_graph, proposals):
    """
    Sorts proposals by length.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph that proposals were generated from.
    proposals : list[frozenset]
        List of proposals.

    Returns
    -------
    list[frozenset]
        Sorted proposals.

    """
    idxs = np.argsort([fragments_graph.proposal_length(p) for p in proposals])
    return [proposals[idx] for idx in idxs]


# --- Batch Formation ---
def get_large_proposal_components(fragments_graph, k):
    flagged_proposals = set()
    visited = set()
    for p in fragments_graph.list_proposals():
        if p not in visited:
            component = fragments_graph.proposal_connected_component(p)
            if len(component) > k:
                flagged_proposals = flagged_proposals.union(component)
            visited = visited.union(component)
    return flagged_proposals
