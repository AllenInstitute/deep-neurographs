"""
Created on Sat Sept 16 11:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

This script executes the full GraphTrace inference pipeline for processing
neuron segmentation data. It performs the following steps:

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
from datetime import datetime
from time import time

import os

from deep_neurographs.intake import GraphBuilder
from deep_neurographs.graph_artifact_removal import remove_doubles
from deep_neurographs.machine_learning.inference import InferenceEngine
from deep_neurographs.utils import io_util, util


class GraphTracePipeline:
    """
    Class that executes the full GraphTrace inference pipeline.

    """
    def __init__(
        self, dataset, pred_id, img_path, model_path, output_dir, config,
    ):
        """
        Initializes an object that executes the full GraphTrace inference
        pipeline.

        Parameters
        ----------
        dataset : int
            Identifier for the dataset to be used in the inference pipeline.
        pred_id : str
            Identifier for the predicted segmentation to be processed by the
            inference pipeline.
        img_path : str
            Path to the raw image of whole brain stored on a GCS bucket.
        model_path : str
            Path to machine learning model parameters.
        output_dir : str
            Directory where the results of the inference will be saved.
        config : Config
            Configuration object containing parameters and settings required
            for the inference pipeline.

        Returns
        -------
        None

        """
        # Class attributes
        self.dataset = dataset
        self.pred_id = pred_id
        self.img_path = img_path
        self.model_path = model_path

        # Extract config settings
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

        # Set output directory
        date = datetime.today().strftime('%Y-%m-%d')
        self.output_dir = f"{output_dir}/{pred_id}-{date}"
        util.mkdir(self.output_dir, delete=True)

    def run(self, fragments_pointer):
        """
        Executes the full inference pipeline.

        Parameters
        ----------
        fragments_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentsGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        None

        """
        # Initializations
        print("\nExperiment Details")
        print("---------------------------------------------------")
        print("Dataset:", self.dataset)
        print("google_pred_id:", self.pred_id)
        print("")
        t0 = time()

        self.build_graph(fragments_pointer)
        self.generate_proposals()
        self.run_inference()
        self.save_results()

        t, unit = util.time_writer(time() - t0)
        print(f"Total Runtime: {round(t, 4)} {unit}\n")

    def build_graph(self, fragments_pointer):
        """
        Initializes and constructs the fragments graph based on the provided
        fragment data.

        Parameters
        ----------
        fragment_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentsGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        None

        """
        print("Building FragmentsGraph...")
        t0 = time()

        # Initialize Graph
        graph_builder = GraphBuilder(
            anisotropy=self.graph_config.anisotropy,
            min_size=self.graph_config.min_size,
            node_spacing=self.graph_config.node_spacing,
            progress_bar=True,
            trim_depth=self.graph_config.trim_depth,
        )
        self.fragments_graph = graph_builder.run(fragments_pointer)

        # Remove doubles (if applicable)
        if self.graph_config.remove_doubles_bool:
            remove_doubles(
                self.fragments_graph, 200, self.graph_config.node_spacing
            )

        # Save valid labels and current graph
        valid_labels_path = os.path.join(self.output_dir, "valid_labels.txt")
        self.fragments_graph.save_valid_labels(valid_labels_path)

        swcs_path = os.path.join(self.output_dir, "processed-swcs.zip")
        self.fragments_graph.to_zipped_swcs(swcs_path)

        t, unit = util.time_writer(time() - t0)
        print(f"Module Runtime: {round(t, 4)} {unit}\n")

    def generate_proposals(self):
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
        print("Generate Proposals...")
        t0 = time()
        self.fragments_graph.generate_proposals(
            self.graph_config.search_radius,
            complex_bool=self.graph_config.complex_bool,
            long_range_bool=self.graph_config.long_range_bool,
            proposals_per_leaf=self.graph_config.proposals_per_leaf,
            trim_endpoints_bool=self.graph_config.trim_endpoints_bool,
        )
        self.fragments_graph.xyz_to_edge = dict()
        n_proposals = util.reformat_number(self.fragments_graph.n_proposals())

        t, unit = util.time_writer(time() - t0)
        print("# Proposals:", n_proposals)
        print(f"Module Runtime: {round(t, 4)} {unit}\n")

    def run_inference(self):
        """
        Executes the inference process using the configured inference engine
        and updates the fragments_graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        print("Run Inference...")
        t0 = time()
        inference_engine = InferenceEngine(
            self.img_path,
            self.model_path,
            self.ml_config.model_type,
            self.graph_config.search_radius,
            confidence_threshold=self.ml_config.threshold,
            downsample_factor=self.ml_config.downsample_factor,
        )
        self.fragments_graph, self.proposal_preds = inference_engine.run(
            self.fragments_graph,
            self.fragments_graph.list_proposals(),
        )

        t, unit = util.time_writer(time() - t0)
        print(f"Module Runtime: {round(t, 4)} {unit}\n")

    def save_results(self):
        print("Save Predictions...")
        t0 = time()
        io_util.save_prediction(
            self.fragments_graph,
            self.proposal_preds,
            self.output_dir,
        )
        t, unit = util.time_writer(time() - t0)
        print(f"Module Runtime: {round(t, 4)} {unit}\n")

    def write_metadata(self):
        metadata = {
            "date": self.date,
            "dataset": self.dataset,
            "pred_id": self.pred_id,
            "min_fragment_input_size": f"{self.graph_config.min_size}um",
            "min_fragment_output_size": f"{self.graph_config.min_size}um",
            "model_type": self.ml_config.model_type,
            "model_name": self.model_name,
            "complex_proposals": self.graph_config.complex_bool,
            "long_range_bool": self.graph_config.long_range_bool,
            "proposals_per_leaf": self.graph_config.proposals_per_leaf,
            "search_radius": f"{self.graph_config.search_radius}um",
            "confidence_threshold": self.ml_config.threshold,
            "node_spacing": self.graph_config.node_spacing,
            "remove_doubles": self.graph_config.remove_doubles_bool,
            "trim_depth": self.graph_config.trim_depth,
        }
        path = os.path.join(self.output_dir, "metadata.json")
        util.write_json(path, metadata)
