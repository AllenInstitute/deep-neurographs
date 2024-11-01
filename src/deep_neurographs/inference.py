"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference with a machine model that classifies edge
proposals.

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
from deep_neurographs.machine_learning.feature_generation import (
    FeatureGenerator,
)
from deep_neurographs.utils import gnn_util
from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import ml_util, util
from deep_neurographs.utils.gnn_util import toCPU
from deep_neurographs.utils.graph_util import GraphLoader

BATCH_SIZE = 2000
CONFIDENCE_THRESHOLD = 0.7


class InferencePipeline:
    """
    Class that executes the full GraphTrace inference pipeline that performs
    the following steps:

    1. Graph Construction
        Builds a graph representation from fragmented neuron segments.

    2. Connection Proposals
        Generates proposals for potential connections between fragments.

    3. Feature Generation
        Extracts relevant features from the proposals and graph to be used by
        a machine learning model.

    4. Inference
        Applies a machine learning model classify proposals as accept/reject
        based on the learned features.

    5. Graph Update
        Integrates the inference results to refine and merge the fragments
        into a cohesive structure.

    """

    def __init__(
        self,
        sample_id,
        segmentation_id,
        img_path,
        model_path,
        output_dir,
        config,
        device=None,
        is_multimodal=False,
        label_path=None,
        log_runtimes=True,
        save_to_s3_bool=False,
        s3_dict=None,
    ):
        """
        Initializes an object that executes the full GraphTrace inference
        pipeline.

        Parameters
        ----------
        sample_id : int
            Identifier for the brain sample to be used in the inference
            pipeline.
        segmentation_id : str
            Identifier for the predicted segmentation to be processed by the
            inference pipeline.
        img_path : str
            Path to the raw image assumed to be stored in a GCS bucket.
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
        label_path : str, optional
            Path to the segmentation assumed to be stored on a GCS bucket. The
            default is None.
        log_runtimes : bool, optional
            Indication of whether to log runtimes. The default is True.
        save_to_s3_bool : bool, optional
            Indication of whether to save result to s3. The default is False.
        s3_dict : dict, optional
            ...

        Returns
        -------
        None

        """
        # Class attributes
        self.accepted_proposals = list()
        self.log_runtimes = log_runtimes
        self.model_path = model_path
        self.sample_id = sample_id
        self.segmentation_id = segmentation_id
        self.save_to_s3_bool = save_to_s3_bool
        self.s3_dict = s3_dict

        # Extract config settings
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

        # Inference engine
        self.inference_engine = InferenceEngine(
            img_path,
            self.model_path,
            self.ml_config.model_type,
            self.graph_config.search_radius,
            confidence_threshold=self.ml_config.threshold,
            device=device,
            downsample_factor=self.ml_config.downsample_factor,
            label_path=label_path,
            is_multimodal=is_multimodal,
        )

        # Set output directory
        self.output_dir = output_dir
        util.mkdir(self.output_dir, delete=True)
        if self.log_runtimes:
            log_path = os.path.join(self.output_dir, "runtimes.txt")
            self.log_handle = open(log_path, 'a')

    # --- Core ---
    def run(self, fragments_pointer):
        """
        Executes the full inference pipeline.

        Parameters
        ----------
        fragments_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        None

        """
        # Initializations
        self.report_experiment()
        self.write_metadata()
        t0 = time()

        # Main
        self.build_graph(fragments_pointer)
        self.generate_proposals()
        self.run_inference()
        self.save_results()

        # Finish
        self.report("Final Graph...")
        self.report_graph()

        t, unit = util.time_writer(time() - t0)
        self.report(f"Total Runtime: {round(t, 4)} {unit}\n")

    def run_schedule(
        self, fragments_pointer, radius_schedule, save_all_rounds=False
    ):
        t0 = time()
        self.report_experiment()
        self.build_graph(fragments_pointer)
        for round_id, radius in enumerate(radius_schedule):
            self.report(f"--- Round {round_id + 1}:  Radius = {radius} ---")
            round_id += 1
            self.generate_proposals(radius)
            self.run_inference()
            if save_all_rounds:
                self.save_results(round_id=round_id)

        if not save_all_rounds:
            self.save_results(round_id=round_id)

        t, unit = util.time_writer(time() - t0)
        self.report(f"Total Runtime: {round(t, 4)} {unit}\n")

    def build_graph(self, fragments_pointer):
        """
        Initializes and constructs the fragments graph based on the provided
        fragment data.

        Parameters
        ----------
        fragment_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        None

        """
        self.report("(1) Building FragmentGraph")
        t0 = time()

        # Initialize Graph
        graph_builder = GraphLoader(
            anisotropy=self.graph_config.anisotropy,
            min_size=self.graph_config.min_size,
            node_spacing=self.graph_config.node_spacing,
            progress_bar=True,
            prune_depth=self.graph_config.prune_depth,
        )
        self.graph = graph_builder.run(fragments_pointer)
        self.filter_fragments()

        # Save valid labels and current graph
        swcs_path = os.path.join(self.output_dir, "processed-swcs.zip")
        labels_path = os.path.join(self.output_dir, "valid_labels.txt")
        n_saved = self.graph.to_zipped_swcs(swcs_path, min_size=100)
        self.graph.save_labels(labels_path)
        self.report(f"# SWCs Saved: {n_saved}")

        # Report runtime
        t, unit = util.time_writer(time() - t0)
        self.report(f"Module Runtime: {round(t, 4)} {unit}")

        # Report graph overview
        self.report("\nInitial Graph...")
        self.report_graph()

    def filter_fragments(self):
        # Filter curvy fragments
        n_curvy = fragment_filtering.remove_curvy(self.graph, 200)
        n_curvy = util.reformat_number(n_curvy)

        # Filter doubles
        if self.graph_config.remove_doubles_bool:
            n_doubles = fragment_filtering.remove_doubles(
                self.graph, 200, self.graph_config.node_spacing
            )
            n_doubles = util.reformat_number(n_doubles)
            self.report(f"# Double Fragments Deleted: {n_doubles}")
        self.report(f"# Curvy Fragments Deleted: {n_curvy}")

    def generate_proposals(self, radius=None):
        """
        Generates proposals for the fragment graph based on the specified
        configuration.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Initializations
        self.report("(2) Generate Proposals")
        if radius is None:
            radius = self.graph_config.search_radius

        # Main
        t0 = time()
        n_trimmed = self.graph.generate_proposals(
            radius,
            complex_bool=self.graph_config.complex_bool,
            long_range_bool=self.graph_config.long_range_bool,
            proposals_per_leaf=self.graph_config.proposals_per_leaf,
            trim_endpoints_bool=self.graph_config.trim_endpoints_bool,
        )

        n_proposals = util.reformat_number(self.graph.n_proposals())
        n_trimmed = util.reformat_number(n_trimmed)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.report(f"# Trimmed: {n_trimmed}")
        self.report(f"# Proposals: {n_proposals}")
        self.report(f"Module Runtime: {round(t, 4)} {unit}\n")

    def run_inference(self):
        """
        Executes the inference process using the configured inference engine
        and updates the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.report("(3) Run Inference")
        t0 = time()
        n_proposals = max(self.graph.n_proposals(), 1)
        self.graph, accepts = self.inference_engine.run(
            self.graph, self.graph.list_proposals()
        )
        self.accepted_proposals.extend(accepts)
        self.report(f"# Accepted: {util.reformat_number(len(accepts))}")
        self.report(f"% Accepted: {round(len(accepts) / n_proposals, 4)}")

        t, unit = util.time_writer(time() - t0)
        self.report(f"Module Runtime: {round(t, 4)} {unit}\n")

    def save_results(self, round_id=None):
        """
        Saves the processed results from running the inference pipeline,
        namely the corrected swc files and a list of the merged swc ids.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Save result locally
        suffix = f"-{round_id}" if round_id else ""
        filename = f"corrected-processed-swcs{suffix}.zip"
        path = os.path.join(self.output_dir, filename)
        self.graph.to_zipped_swcs(path)
        self.save_connections(round_id=round_id)
        self.write_metadata()

        # Save result on s3
        if self.save_to_s3_bool:
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
        # Initializations
        bucket_name = self.s3_dict["bucket_name"]
        date = datetime.today().strftime("%Y%m%d")
        subdir_name = f"/corrected_{self.sample_id}_{self.segmentation_id}_{date}"
        prefix = self.s3_dict["prefix"] + subdir_name

        # Move result files
        for filename in os.listdir(self.output_dir):
            if filename != "processed-swcs.zip":
                local_path = os.path.join(self.output_dir, filename)
                s3_path = os.path.join(prefix, filename)
                util.write_to_s3(local_path, bucket_name, s3_path)
        print("Results written to S3 prefix -->", prefix)

    # --- io ---
    def save_connections(self, round_id=None):
        """
        Saves predicted connections between connected components in a txt file.

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
            "sample_id": self.sample_id,
            "segmentation_id": self.segmentation_id,
            "min_fragment_size": f"{self.graph_config.min_size}um",
            "model_type": self.ml_config.model_type,
            "model_name": os.path.basename(self.model_path),
            "complex_proposals": self.graph_config.complex_bool,
            "long_range_bool": self.graph_config.long_range_bool,
            "proposals_per_leaf": self.graph_config.proposals_per_leaf,
            "search_radius": f"{self.graph_config.search_radius}um",
            "confidence_threshold": self.ml_config.threshold,
            "node_spacing": self.graph_config.node_spacing,
            "remove_doubles": self.graph_config.remove_doubles_bool,
        }
        path = os.path.join(self.output_dir, "metadata.json")
        util.write_json(path, metadata)

    # --- Summaries ---
    def report(self, txt):
        print(txt)
        if self.log_runtimes:
            self.log_handle.write(txt)
            self.log_handle.write("\n")

    def report_experiment(self):
        self.report("\nExperiment Overview")
        self.report("-------------------------------------------------------")
        self.report(f"Sample_ID: {self.sample_id}")
        self.report(f"Segmentation_ID: {self.segmentation_id}")
        self.report("\n")

    def report_graph(self):
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
        self.report(f"# Connected Components: {n_components}")
        self.report(f"# Nodes: {n_nodes}")
        self.report(f"# Edges: {n_edges}")
        self.report(f"Memory Consumption: {usage} GBs\n")


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
        radius,
        batch_size=BATCH_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=None,
        downsample_factor=1,
        label_path=None,
        is_multimodal=False
    ):
        """
        Initializes an inference engine by loading images and setting class
        attributes.

        Parameters
        ----------
        img_path : str
            Path to image stored in a GCS bucket.
        model_path : str
            Path to machine learning model parameters.
        model_type : str
            Type of machine learning model used to perform inference.
        radius : float
            Search radius used to generate proposals.
        batch_size : int, optional
            Number of proposals to generate features and classify per batch.
            The default is the global varaible "BATCH_SIZE".
        confidence_threshold : float, optional
            Threshold on acceptance probability for proposals. The default is
            the global variable "CONFIDENCE_THRESHOLD".
        downsample_factor : int, optional
            Downsampling factor that accounts for which level in the image
            pyramid the voxel coordinates must index into. The default is 0.

        Returns
        -------
        None

        """
        # Set class attributes
        self.batch_size = batch_size
        self.device = "cpu" if device is None else device
        self.is_gnn = True if "Graph" in model_type else False
        self.radius = radius
        self.threshold = confidence_threshold

        # Features
        self.feature_generator = FeatureGenerator(
            img_path,
            downsample_factor,
            label_path=label_path,
            is_multimodal=is_multimodal
        )

        # Model
        self.model = ml_util.load_model(model_path)
        if self.is_gnn:
            self.model = self.model.to(self.device)
            self.model.eval()

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
        # Initializations
        assert not gutil.cycle_exists(neurograph), "Graph contains cycle!"
        if self.is_gnn:
            proposals = set(proposals)
        else:
            proposals = sort_proposals(neurograph, proposals)

        # Main
        with tqdm(total=len(proposals), desc="Inference") as pbar:
            accepts = list()
            while len(proposals) > 0:
                # Predict
                batch = self.get_batch(neurograph, proposals)
                dataset = self.get_batch_dataset(neurograph, batch)
                preds = self.predict(dataset)

                # Update graph
                for p in get_accepts(neurograph, preds, self.threshold):
                    neurograph.merge_proposal(p)
                    accepts.append(p)
                pbar.update(len(batch["proposals"]))
            neurograph.absorb_reducibles()
        return neurograph, accepts

    def get_batch(self, neurograph, proposals):
        """
        Generates a batch of proposals.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that proposals were generated from.
        proposals : list
            Proposals for which batch is to be generated from.

        Returns
        -------
        dict
            Batch which consists of a subset of "proposals" and the
            computation graph if the model type is a gnn.

        """
        if self.is_gnn:
            return gnn_util.get_batch(neurograph, proposals, self.batch_size)
        else:
            batch = {"proposals": proposals[0:self.batch_size], "graph": None}
            del proposals[0:self.batch_size]
            return batch

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
        features = self.feature_generator.run(neurograph, batch, self.radius)
        computation_graph = batch["graph"] if type(batch) is dict else None
        dataset = ml_util.init_dataset(
            neurograph,
            features,
            self.is_gnn,
            computation_graph=computation_graph,
        )
        return dataset

    def predict(self, dataset):
        """
        Runs the model on the given dataset to generate and filter
        predictions.

        Parameters
        ----------
        data : ...
            Dataset on which the model inference is to be run.

        Returns
        -------
        dict
            A dictionary that maps a proposal to the model's prediction (i.e.
            probability).

        """
        # Get predictions
        if self.is_gnn:
            with torch.no_grad():
                # Get inputs
                n = len(dataset.data["proposal"]["y"])
                x, edge_index, edge_attr = gnn_util.get_inputs(
                    dataset.data, device=self.device
                )

                # Run model
                preds = sigmoid(self.model(x, edge_index, edge_attr))
                preds = toCPU(preds[0:n, 0])
        else:
            preds = np.array(self.model.predict_proba(dataset.data.x)[:, 1])

        # Reformat prediction
        idxs = dataset.idxs_proposals["idx_to_id"]
        return {idxs[i]: p for i, p in enumerate(preds)}


# --- Accepting Proposals ---
def get_accepts(neurograph, preds, threshold, high_threshold=0.9):
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
    preds = {k: v for k, v in preds.items() if v > threshold}
    best_proposals, proposals = separate_best(
        preds, neurograph.simple_proposals(), high_threshold
    )

    # Determine which proposals to accept
    accepts = list()
    accepts.extend(filter_proposals(neurograph, best_proposals))
    accepts.extend(filter_proposals(neurograph, proposals))
    neurograph.remove_edges_from(map(tuple, accepts))
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


def sort_proposals(neurograph, proposals):
    """
    Sorts proposals by length.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that proposals were generated from.
    proposals : list[frozenset]
        List of proposals.

    Returns
    -------
    list[frozenset]
        Sorted proposals.

    """
    idxs = np.argsort([neurograph.proposal_length(p) for p in proposals])
    return [proposals[idx] for idx in idxs]
