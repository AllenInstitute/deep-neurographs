"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that executes the full GraphTrace inference pipeline.

    Inference Algorithm:
        1. Graph Construction
            Build graph from neuron fragments.

        2. Proposal Generation
            Generate proposals for potential connections between fragments.

        3. Proposal Classification
            a. Feature Generation
                Extract features from proposals and graph for a machine
                learning model.
            b. Predict with Graph Neural Network (GNN)
                Run a GNN to classify proposals as accept/reject
                based on the learned features.
            c. Merge Accepted Proposals
                Add accepted proposals to the fragments graph as edges.

Note: Steps 2 and 3 of the inference pipeline can be iterated in a loop that
      repeats multiple times by calling the routine "run_schedule" within the
      InferencePipeline class.

"""


from collections import deque
from datetime import datetime
from time import time
from torch.nn.functional import sigmoid
from tqdm import tqdm

import ast
import networkx as nx
import numpy as np
import os
import torch

from deep_neurographs import fragment_filtering
from deep_neurographs.fragments_graph import FragmentsGraph
from deep_neurographs.machine_learning import datasets
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
        self.img_path = img_path
        self.model_path = model_path
        self.brain_id = brain_id
        self.segmentation_id = segmentation_id
        self.segmentation_path = segmentation_path
        self.somas_path = somas_path
        self.s3_dict = s3_dict

        # Extract config settings
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

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
        self.connect_soma_fragments() if self.somas_path else None
        self.generate_proposals()
        self.classify_proposals()

        # Finish
        t, unit = util.time_writer(time() - t0)
        self.report_graph(prefix="\nFinal")
        self.report(f"Total Runtime: {round(t, 4)} {unit}\n")
        self.save_results()

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

        # Finish
        t, unit = util.time_writer(time() - t0)
        self.report_graph(prefix="\nFinal")
        self.report(f"Total Runtime: {round(t, 4)} {unit}\n")
        self.save_results()

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

        print("# Soma Fragments:", len(self.graph.soma_ids))

        # Save valid labels and current graph
        swc_dir = os.path.join(self.output_dir, "swcs")
        valid_labels_path = os.path.join(self.output_dir, "valid_labels.txt")
        self.graph.to_zipped_swcs(swc_dir, sampling_rate=2)
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

    def connect_soma_fragments(self):
        self.graph.init_kdtree()
        nodes_list = list()
        merge_cnt, soma_cnt = 0, 0
        for soma_xyz in map(ast.literal_eval, util.read_txt(self.somas_path)):
            hits = self.graph.find_fragments_near_xyz(soma_xyz, 20)
            if len(hits) > 1:
                # Determine new swc id
                soma_cnt += 1
                hit_soma = None
                for swc_id in hits:
                    if swc_id in self.graph.soma_ids and hit_soma:
                        break
                self.graph.soma_ids.add(swc_id)

                # Add soma node
                soma_node = self.graph.node_cnt + 1
                self.graph.add_node(
                    soma_node,
                    proposals=set(),
                    radius=2,
                    swc_id=swc_id,
                    xyz=soma_xyz,
                )
                self.graph.node_cnt += 1

                # Merge fragments to soma
                for swc_id_i, i in hits.items():
                    radius = np.array([2, 2])
                    xyz_i = self.graph.nodes[i]["xyz"]
                    xyz = np.array([soma_xyz, xyz_i])
                    self.graph.add_edge(soma_node, i, radius=radius, xyz=xyz)
                    self.graph.xyz_to_edge[tuple(xyz_i)] = (soma_node, i)
                    self.graph.upd_ids(swc_id, i)
                    merge_cnt += 1

        print("# Somas Connected:", soma_cnt)
        print("# Merges:", merge_cnt)
        del self.graph.kdtree

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
        self.report(f"# Proposals Blocked: {self.graph.n_proposals_blocked}")
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
        self.inference_engine = InferenceEngine(
            self.graph,
            self.img_path,
            self.model_path,
            self.ml_config,
            self.graph_config.search_radius,
            segmentation_path=self.segmentation_path,
        )
        accepts = self.inference_engine.run()
        self.accepted_proposals.extend(accepts)

        # Report results
        t, unit = util.time_writer(time() - t0)
        self.report(f"# Merges Blocked: {self.graph.n_merges_blocked}")
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
        self.graph.to_zipped_swcs(swc_dir)
        self.save_connections()
        self.write_metadata()
        self.log_handle.close()

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
        results_prefix = self.s3_dict["prefix"] + "/corrected-swcs/"
        print("Results at", f"{bucket_name}/{results_prefix}")

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
        graph,
        img_path,
        model_path,
        ml_config,
        radius,
        segmentation_path=None,
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
        ml_config : MLConfig
            Configuration object containing parameters and settings required
            for the inference.
        radius : float
            Search radius used to generate proposals.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. The default is None.

        Returns
        -------
        None

        """
        # Instance attributes
        self.batch_size = ml_config.batch_size
        self.device = ml_config.device
        self.graph = graph
        self.ml_config = ml_config
        self.radius = radius
        self.threshold = ml_config.threshold

        # Feature generator
        self.feature_generator = FeatureGenerator(
            self.graph,
            img_path,
            anisotropy=self.ml_config.anisotropy,
            is_multimodal=self.ml_config.is_multimodal,
            multiscale=self.ml_config.multiscale,
            segmentation_path=segmentation_path,
        )

        # Model
        if ml_config.is_multimodal:
            self.model = ml_util.init_model(ml_config.is_multimodal)
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model = torch.load(model_path, weights_only=False)
        self.model.to(self.device)
        self.model.eval()

    def init_dataloader(self):
        if len(self.graph.soma_ids) > 0:
            return SeededGraphDataLoader(self.graph, self.batch_size)
        else:
            return GraphDataLoader(self.graph, self.batch_size)

    def run(self, return_preds=False):
        """
        Runs inference by forming batches of proposals, then performing the
        following steps for each batch: (1) generate features, (2) classify
        proposals by running model, and (3) adding each accepted proposal as
        an edge to "graph" if it does not create a cycle.

        Parameters
        ----------
        graph : FragmentsGraph
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
        # Initializations
        dataloader = self.init_dataloader()
        pbar = tqdm(total=self.graph.n_proposals(), desc="Inference")

        # Main
        accepts = list()
        hat_y = dict()
        for batch in dataloader:
            # Feature generation
            features = self.feature_generator.run(batch, self.radius)
            heterograph_data = datasets.init(features, batch["graph"])

            # Run model
            hat_y_i = self.predict(heterograph_data)
            if return_preds:
                hat_y.update(hat_y_i)

            # Determine which proposals to accept
            accepts.extend(self.update_graph(hat_y_i))
            pbar.update(len(batch["proposals"]))

        # Return results
        if return_preds:
            return accepts, hat_y
        else:
            return accepts

    def predict(self, heterograph_data):
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
        # Generate predictions
        with torch.no_grad():
            x, edge_index, edge_attr = ml_util.get_inputs(
                heterograph_data.data, self.device
            )
            hat_y = sigmoid(self.model(x, edge_index, edge_attr))

        # Reformat predictions
        n_proposals = len(heterograph_data.data["proposal"]["y"])
        hat_y = ml_util.toCPU(hat_y[0:n_proposals, 0])
        idxs = heterograph_data.idxs_proposals["idx_to_id"]
        return {idxs[i]: p for i, p in enumerate(hat_y)}

    def update_graph(self, preds, high_threshold=0.9):
        """
        Determines which proposals to accept based on prediction scores and
        the specified threshold.

        Parameters
        ----------
        preds : dict
            Dictionary that maps proposal ids to probability generated from
            machine learning model.
        high_threshold : float, optional
            Threshold value for separating the best proposals from the rest.
            The default is 0.9.

        Returns
        -------
        list
            Proposals to be added as edges to "graph".

        """
        # Partition proposals into best and the rest
        preds = {k: v for k, v in preds.items() if v > self.threshold}
        best_proposals, proposals = self.separate_best(preds, high_threshold)

        # Determine which proposals to accept
        accepts = list()
        accepts.extend(self.add_accepts(best_proposals))
        accepts.extend(self.add_accepts(proposals))
        return accepts

    def separate_best(self, preds, high_threshold):
        """
        Splits "preds" into two separate dictionaries such that one contains
        the best proposals (i.e. simple proposals with high confidence) and
        the other contains all other proposals.

        Parameters
        ----------
        preds : dict
            Dictionary that maps proposal ids to probability generated from
            machine learning model.
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
        simple_proposals = self.graph.simple_proposals()
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

    def add_accepts(self, proposals):
        """
        ...

        Parameters
        ----------
        proposals : list[frozenset]
            Proposals with predicted probability above threshold to be added
            to the graph.

        Returns
        -------
        List[frozenset]
            List of proposals that do not create a cycle when iteratively
            added to "graph".

        """
        accepts = list()
        for proposal in proposals:
            i, j = tuple(proposal)
            if not nx.has_path(self.graph, i, j):
                self.graph.merge_proposal(proposal)
                accepts.append(proposal)
        return accepts


# --- Custom Dataloaders ---
class GraphDataLoader:

    def __init__(self, graph, batch_size=200, gnn_depth=2):
        # Instance attributes
        self.batch_size = batch_size
        self.gnn_depth = gnn_depth
        self.graph = graph
        self.proposals = set(graph.list_proposals())

        # Identify clustered proposals
        self.flagged = set()  # self.find_proposal_clusters(5)

    def find_proposal_clusters(self, k):
        flagged = set()
        visited = set()
        for proposal in self.proposals:
            if proposal not in visited:
                cluster = self.extract_cluster(proposal)
                if len(cluster) >= k:
                    flagged = flagged.union(cluster)
                visited = visited.union(cluster)
        return flagged

    def extract_cluster(self, proposal):
        """
        Extracts the connected component that "proposal" belongs to in the
        proposal induced subgraph.

        Parameters
        ----------
        proposal : Frozenset[int]
            Proposal used as the root to extract its connected component in
            the proposal induced subgraph.

        Returns
        -------
        Set[Frozenset[int]]
            Connected component that "proposal" belongs to in the proposal
            induced subgraph.

        """
        queue = deque([proposal])
        visited = set()
        while len(queue) > 0:
            # Visit proposal
            proposal = queue.pop()
            visited.add(proposal)

            # Update queue
            for i in proposal:
                for j in self.graph.nodes[i]["proposals"]:
                    proposal_ij = frozenset({i, j})
                    if proposal_ij not in visited:
                        queue.append(proposal_ij)
        return visited

    # --- Batch Generation ---
    def __iter__(self):
        while self.proposals:
            # Run BFS
            batch = {"graph": nx.Graph(), "proposals": set()}
            while not self.is_batch_full(batch) and self.proposals:
                root = util.sample_once(self.proposals)
                self.populate_via_bfs(batch, root)

            # Yield batch
            yield batch

    def populate_via_bfs(self, batch, root):
        i, j = tuple(root)
        queue = deque([(i, 0), (j, 0)])
        visited = set({i, j})
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            self.visit_nbhd(batch, i)
            self.visit_proposals(batch, queue, visited, i)

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    n_j = len(self.graph.nodes[j]["proposals"])
                    d_j = min(d_i + 1, -n_j)
                    if d_j <= self.gnn_depth:
                        queue.append((j, d_j))
                        visited.add(j)

    def visit_nbhd(self, batch, i):
        for j in self.graph.neighbors(i):
            batch["graph"].add_edge(i, j)

    def visit_proposals(self, batch, queue, visited, i):
        if not self.is_batch_full(batch):
            for j in self.graph.nodes[i]["proposals"]:
                # Visit proposal
                proposal = frozenset({i, j})
                if proposal in self.proposals:
                    batch["graph"].add_edge(i, j)
                    batch["proposals"].add(proposal)
                    self.proposals.remove(proposal)
                    if j not in visited:
                        queue.append((j, 0))

                # Check if proposal is flagged
                # proposal in self.flagged and proposal in self.proposals:
                if False:
                    self.visit_flagged_proposal(batch)

    def visit_flagged_proposal(self, batch, queue, visited, proposal):
        nodes_added = set()
        for p in self.extract_cluster(proposal):
            # Add proposal
            node_1, node_2 = tuple(p)
            batch["graph"].add_edge(node_1, node_2)
            batch["proposals"].add(p)

            # Update queue
            if not (node_1 in visited and node_1 in nodes_added):
                queue.append((node_1, 0))
            if not (node_2 in visited and node_2 in nodes_added):
                queue.append((node_2, 0))

    def is_batch_full(self, batch):
        return True if len(batch["proposals"]) >= self.batch_size else False


class SeededGraphDataLoader(GraphDataLoader):

    def __init__(self, graph, batch_size=200, gnn_depth=2):
        # Call parent class
        super(SeededGraphDataLoader, self).__init__(
            graph, batch_size, gnn_depth
        )

    # --- Batch Generation ---
    def __iter__(self):
        soma_connected_proposals_exist = True
        while soma_connected_proposals_exist:
            # Run BFS
            batch = {
                "graph": nx.Graph(),
                "proposals": set(),
                "soma_proposals": set()
            }
            while not self.is_batch_full(batch) and self.proposals:
                root = self.find_bfs_root()
                if root:
                    self.populate_via_seeded_bfs(batch, root)
                else:
                    soma_connected_proposals_exist = False
                    break

            # Yield batch
            if batch["proposals"]:
                yield batch

        # Call parent class dataloader
        for batch in super().__iter__():
            yield batch

    def find_bfs_root(self):
        for proposal in self.proposals:
            i, j = tuple(proposal)
            if self.graph.is_soma(i):
                return i
            elif self.graph.is_soma(j):
                return j
        return False

    def populate_via_seeded_bfs(self, batch, root):
        queue = self.init_seeded_queue(root)
        visited = set({root})
        while queue:
            # Visit node
            i, d_i = queue.popleft()
            self.visit_nbhd(batch, i)
            self.visit_proposals_seeded(batch, queue, visited, i)

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    n_j = len(self.graph.nodes[j]["proposals"])
                    d_j = min(d_i + 1, -n_j)
                    if d_j <= self.gnn_depth:
                        queue.append((j, d_j))
                        visited.add(j)

    def init_seeded_queue(self, root):
        seeded_queue = deque([(root, 0)])
        queue = deque([root])
        visited = set({root})
        while queue:
            # Visit node
            i = queue.pop()
            if self.graph.nodes[i]["proposals"]:
                seeded_queue.append((i, 0))

            # Update queue
            for j in self.graph.neighbors(i):
                if j not in visited:
                    queue.append(j)
                    visited.add(j)
        return seeded_queue

    def visit_proposals_seeded(self, batch, queue, visited, i):
        if len(batch["proposals"]) < self.batch_size:
            for j in self.graph.nodes[i]["proposals"]:
                # Visit proposal
                proposal = frozenset({i, j})
                if proposal in self.proposals:
                    batch["graph"].add_edge(i, j)
                    batch["proposals"].add(proposal)
                    self.proposals.remove(proposal)
                    if j not in visited:
                        queue.append((j, 0))

                # Check if proposal is connected to soma
                if self.graph.is_soma(i) or self.graph.is_soma(j):
                    batch["soma_proposals"].add(proposal)
