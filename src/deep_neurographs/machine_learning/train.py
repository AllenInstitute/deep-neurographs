"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that classify proposals.


To do: explain how the train pipeline is organized. how is the data organized?


"""

from collections import defaultdict
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import random
import torch

from deep_neurographs.fragments_graph import FragmentsGraph
from deep_neurographs.machine_learning import datasets
from deep_neurographs.machine_learning.feature_generation import (
    FeatureGenerator
)
from deep_neurographs.utils import ml_util, util


class FragmentsGraphDataset:
    """
    Custom dataset for storing a list of graphs to be used to train a graph
    neural network. Graph are stored in the "self.graphs" attribute, which is
    a dictionary containing the followin items:
        - Key: (brain_id, segmentation_id, example_id)
        - Value: graph that is an instance of FragmentsGraph

    This dataset is populated using the "self.add_graph" method, which
    requires the following inputs:
        (1) key: Unique identifier of graph.
        (2) gt_pointer: Path to ground truth SWC files.
        (2) pred_pointer: Path to predicted SWC files.
        (3) img_path: Path to whole-brain image stored in cloud bucket.
        (4) segmentation_path: Path to whole-brain segmentation stored on GCS.

    Note: This dataset supports graphs from multiple whole-brain datasets.

    """

    def __init__(self, config):
        # Instance Attributes
        self.features = dict()
        self.feature_generators = dict()
        self.graphs = dict()
        self.validation_keys = set()

        # Configs
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

    def init_feature_generator(self, key, img_path, segmentation_path):
        brain_id, segmentation_id, _ = key
        generator_key = (brain_id, segmentation_id)
        if generator_key not in self.feature_generators:
            self.feature_generators[generator_key] = FeatureGenerator(
                img_path,
                self.ml_config.multiscale,
                anisotropy=self.ml_config.anisotropy,
                is_multimodal=self.ml_config.is_multimodal,
                segmentation_path=segmentation_path,
                transform=self.ml_config.transform,
            )

    def __len__(self):
        """
        Counts the number of graphs in self.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of graphs in self.

        """
        return len(self.graphs)

    def n_proposals(self):
        """
        Counts the number of proposals across all graphs in self.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of proposals across all graphs in self.

        """
        return np.sum([graph.n_proposals() for graph in self.graphs.values()])

    def p_accepts(self):
        """
        Computes the percentage of accepted proposals in ground truth.

        Parameters
        ----------
        None

        Returns
        -------
        float
            Percentage of accepted proposals in ground truth.

        """
        n_proposals = self.n_proposals()
        cnt_accepts = [len(graph.gt_accepts) for graph in self.graphs.values()]
        return np.sum(cnt_accepts) / n_proposals

    # --- Load Data ---
    def add_graph(
        self,
        key,
        gt_pointer,
        pred_pointer,
        img_path,
        segmentation_path=None
    ):
        # Add graph
        self.graphs[key] = self.load_graph(pred_pointer)
        self.graphs[key].generate_proposals(
            self.graph_config.search_radius,
            complex_bool=self.graph_config.complex_bool,
            groundtruth_graph=self.load_graph(gt_pointer),
            long_range_bool=self.graph_config.long_range_bool,
            proposals_per_leaf=self.graph_config.proposals_per_leaf,
        )

        # Generate features
        self.features[key] = self.generate_features(
            key, img_path, img_path, segmentation_path
        )

    def load_graph(self, swc_pointer):
        """
        Loads a graph by reading and processing SWC files specified by
        "swc_pointer".

        Parameters
        ----------
        swc_pointer : Any
            Object that points to SWC files to be read, must be one of:
                - swc_dir (str): Path to directory containing SWC files.
                - swc_path (str): Path to single SWC file.
                - swc_path_list (List[str]): List of paths to SWC files.
                - swc_zip (str): Path to a ZIP archive containing SWC files.
                - gcs_dict (dict): Dictionary that contains the keys
                  "bucket_name" and "path" to read from a GCS bucket.

        Returns
        -------
        FragmentsGraph
            Graph constructed from SWC files.

        """
        graph = FragmentsGraph(
            min_size=self.graph_config.min_size,
            node_spacing=self.graph_config.node_spacing,
        )
        graph.load_fragments(swc_pointer)
        return graph

    def generate_features(self, key, graph, img_path, segmentation_path):
        # Initializations
        self.init_feature_generator(key, img_path, segmentation_path)
        proposals_dict = {
            "proposals": self.graphs[key].list_proposals(),
            "graph": self.graphs[key].copy_graph()
        }

        # Main
        generator_key = (key[0], key[1])
        features = self.feature_generators[generator_key].run(
            self.graphs[key],
            proposals_dict,
            self.graph_config.search_radius
        )
        return features

    def generate_batches(self, batch_size):
        """
        Generates a list of batches for training a graph neural network. Each
        batch is a tuple that contains the following:
            - key (str): Unique identifier of a graph in self.graphs.
            - computation_graph (networkx.Graph): GNN computation graph.
            - proposals (List[frozenset[int]]): List of proposals in graph.

        Parameters
        ----------
        batch_size : int
            Maximum number of proposals in each batch.

        Returns
        -------
        List[tuple]
            List of batches for training a graph neural network.

        """
        # Initializations
        keys = list(self.graphs.keys())
        random.shuffle(keys)

        # Main
        batches = list()
        for key in keys:
            proposals = set(self.graphs[key].list_proposals())
            while len(proposals) > 0:
                # Get batch
                batch = ml_util.get_batch(
                    self.graphs[key], proposals, batch_size
                )
                proposals -= batch["proposals"]

                # Add dataset
                accepts = self.graphs[key].gt_accepts
                features = self.extract_features(key, batch)
                dataset = datasets.init(features, batch["graph"], accepts)
                batches.append((key, dataset))
        return batches

    def extract_validation(self, batch_size, validation_percent):
        # Initializations
        n_proposals = self.n_proposals() * validation_percent
        batches = self.generate_batches(batch_size)
        validation_batches = list()
        keys = set()

        # Populate validation set
        cnt = 0
        while cnt < n_proposals:
            key, dataset = batches.pop()
            validation_batches.append((key, dataset))
            cnt += dataset.n_proposals()
            keys.add(key)

        # Add batches from same graphs
        while len(batches) > 0:
            key, dataset = batches.pop()
            if key in keys:
                validation_batches.append((key, dataset))

        # Delete graphs in validation set
        for key in keys:
            del self.graphs[key]
        return validation_batches, cnt

    # --- Helpers ---
    def extract_features(self, key, batch):
        # Node features
        features = defaultdict(lambda: defaultdict(dict))
        for i in batch["graph"].nodes:
            features["nodes"][i] = self.features[key]["nodes"][i]

        # Edge features
        for e in map(frozenset, batch["graph"].edges):
            if e in batch["proposals"]:
                features["proposals"][e] = self.features[key]["proposals"][e]
            else:
                features["branches"][e] = self.features[key]["branches"][e]

        # Image patches
        if self.ml_config.is_multimodal:
            for p in batch["proposals"]:
                features["patches"][p] = self.features[key]["patches"][p]
        return features


class Trainer:
    """
    Custom class that trains graph neural networks.

    """

    def __init__(
        self,
        output_dir,
        batch_size=64,
        device="cuda",
        lr=1e-4,
        n_epochs=1000,
        validation_percent=0.15
    ):
        """
        Constructs a GraphTrainer object.

        Parameters
        ----------
        n_epochs : int
            Number of epochs. The default is 1000.
        ...

        Returns
        -------
        None.

        """
        # Initializations
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, "tensorboards", exp_name)
        model_dir = os.path.join(output_dir, "saved-models", exp_name)
        util.mkdir(output_dir)
        util.mkdir(model_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.model_dir = model_dir
        self.validation_percent = validation_percent
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self, model, graph_dataset):
        """
        Trains a graph neural network in the case where "datasets" is a
        dictionary of datasets such that each corresponds to a distinct graph.

        Parameters
        ----------
        ...

        Returns
        -------
        torch.nn.Module
            Graph neural network that has been fit onto the given graph
            dataset.

        """
        # Initializations
        model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)

        # Validation data
        validation_batches, example_cnt = graph_dataset.extract_validation(
            self.batch_size, self.validation_percent
        )
        self.save_validation_keys(validation_batches)
        print("# Train Examples:", graph_dataset.n_proposals())
        print("# Validation Examples:", example_cnt)

        # Main
        best_f1 = 0
        n_upds = 0
        for epoch in range(self.n_epochs):
            # Train
            y, hat_y = [], []
            model.train()
            for _, dataset in graph_dataset.generate_batches(self.batch_size):
                # Forward pass
                hat_y_i, y_i = self.predict(model, dataset.data)
                loss = self.criterion(hat_y_i, y_i)
                self.writer.add_scalar("loss", loss, n_upds)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                n_upds += 1

                # Store prediction
                y.extend(ml_util.toCPU(y_i))
                hat_y.extend(ml_util.toCPU(hat_y_i))

            train_f1 = self.compute_metrics(y, hat_y, "train", epoch)
            self.scheduler.step()

            # Validate
            model.eval()
            y, hat_y = [], []
            for _, dataset in validation_batches:
                hat_y_i, y_i = self.predict(model, dataset.data)
                y.extend(ml_util.toCPU(y_i))
                hat_y.extend(ml_util.toCPU(hat_y_i))

            # Check for new best model
            val_f1 = self.compute_metrics(y, hat_y, "val", epoch)
            scores = f"Epoch {epoch}:  train_f1={train_f1}  val_f1={val_f1}"
            if val_f1 > best_f1:
                print(scores + "  --  New Best!")
                best_f1 = val_f1
                self.save_model(model, best_f1)
            else:
                print(scores)

    def predict(self, model, data):
        """
        Runs "data" through "self.model" to generate a prediction.

        Parameters
        ----------
        data : GraphDataset
            Graph dataset that corresponds to a single connected component.

        Returns
        -------
        torch.Tensor
            Ground truth.
        torch.Tensor
            Prediction.

        """
        x, edge_index, edge_attr = ml_util.get_inputs(data, self.device)
        hat_y = model(x, edge_index, edge_attr)
        y = data["proposal"]["y"]
        return truncate(hat_y, y), y

    def compute_metrics(self, y, hat_y, prefix, epoch):
        """
        Computes and logs evaluation metrics for binary classification.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth.
        hat_y : torch.Tensor
            Prediction.
        prefix : str
            Prefix to be added to the metric names when logging.
        epoch : int
            Current epoch.

        Returns
        -------
        float
            F1 score.

        """
        # Initializations
        y = np.array(y, dtype=int).tolist()
        hat_y = (np.array(hat_y) > 0).tolist()

        # Compute
        accuracy = accuracy_score(y, hat_y)
        accuracy_dif = accuracy - np.sum(y) / len(y)
        precision = precision_score(y, hat_y)
        recall = recall_score(y, hat_y)
        f1 = f1_score(y, hat_y)

        # Log
        self.writer.add_scalar(prefix + "_accuracy:", accuracy, epoch)
        self.writer.add_scalar(prefix + "_accuracy_df:", accuracy_dif, epoch)
        self.writer.add_scalar(prefix + "_precision:", precision, epoch)
        self.writer.add_scalar(prefix + "_recall:", recall, epoch)
        self.writer.add_scalar(prefix + "_f1:", f1, epoch)
        return round(f1, 4)

    def save_model(self, model, score):
        date = datetime.today().strftime("%Y%m%d")
        filename = f"GraphNeuralNet-{date}-{round(score, 4)}.pth"
        path = os.path.join(self.model_dir, filename)
        torch.save(model.state_dict(), path)

    def save_validation_keys(self, validation_batches):
        keys = set([key for key, _ in validation_batches])
        path = os.path.join(self.model_dir, "validation-examples.txt")
        util.write_list(path, keys)


# -- Helpers --
def truncate(hat_y, y):
    """
    Truncates "hat_y" so that this tensor has the same shape as "y". Note this
    operation removes the predictions corresponding to branches so that loss
    is computed over proposals.

    Parameters
    ----------
    hat_y : torch.Tensor
        Tensor to be truncated.
    y : torch.Tensor
        Tensor used as a reference.

    Returns
    -------
    torch.Tensor
        Truncated "hat_y".

    """
    return hat_y[: y.size(0), 0]
