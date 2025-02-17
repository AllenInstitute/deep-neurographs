"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that classify proposals.


To do: explain how the train pipeline is organized. how is the data organized?


"""


from copy import deepcopy
from datetime import datetime
from random import sample, shuffle
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
import os
import torch

from deep_neurographs.fragments_graph import FragmentsGraph
from deep_neurographs.machine_learning import heterograph_datasets
from deep_neurographs.machine_learning.feature_generation import (
    FeatureGenerator,
)
from deep_neurographs.utils import gnn_util, ml_util


class TrainPipeline:
    """
    Class that is used to train a machine learning model that classifies
    proposals.

    """
    def __init__(
        self,
        config,
        model,
        output_dir,
        criterion=None,
        is_multimodal=False,
        validation_ids=None,
    ):
        # Instance attributes
        self.feature_generators = dict()
        self.idx_to_ids = list()
        self.output_dir = output_dir
        self.is_multimodal = is_multimodal

        # Config Parameters
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config

        # Data Structures for Examples
        self.gt_graphs = list()
        self.pred_graphs = list()
        self.train_dataset_list = list()
        self.validation_dataset_list = list()

        # Machine Learning
        self.criterion = criterion if criterion else BCEWithLogitsLoss()
        self.model = model
        self.validation_ids = validation_ids

    # --- getters/setters ---
    def n_graphs(self):
        """
        Counts the number of graphs loaded into self.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of graphs loaded into self.

        """
        return len(self.gt_graphs)

    def n_train_graphs(self):
        """
        Counts the number of graphs in the training set.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of graphs in the training set.

        """
        return len(self.train_dataset_list)

    def n_validation_graphs(self):
        """
        Counts the number of graphs in the validation set.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of graphs in the training set.

        """
        return len(self.validation_dataset_list)

    def set_validation_idxs(self):
        if self.validation_ids is None:
            k = int(self.ml_config.validation_split * self.n_graphs())
            self.validation_idxs = sample(np.arange(self.n_graphs), k)
        else:
            self.validation_idxs = list()
            for ids in self.validation_ids:
                for i in range(self.n_graphs()):
                    same = all([ids[k] == self.idx_to_ids[i][k] for k in ids])
                    if same:
                        self.validation_idxs.append(i)
        assert len(self.validation_idxs) > 0, "No validation data!"

    # --- loaders ---
    def load_example(
        self,
        gt_pointer,
        pred_pointer,
        brain_id,
        example_id=None,
        segmentation_id=None,
    ):
        # Load graphs
        self.gt_graphs.append(self.load_graph(gt_pointer))
        self.pred_graphs.append(self.load_graph(pred_pointer))

        # Set example ids
        self.idx_to_ids.append(
            {
                "brain_id": brain_id,
                "example_id": example_id,
                "segmentation_id": segmentation_id,
            }
        )

    def load_graph(self, swc_pointer):
        """
        Loads a graph by reading and processing SWC files specified by
        "swc_pointer".

        Parameters
        ----------
        swc_pointer : dict / list / str
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
            Graph that is constructed from SWC files.

        """
        graph = FragmentsGraph(
            min_size=self.graph_config.min_size,
            node_spacing=self.graph_config.node_spacing,
        )
        graph.load_fragments(swc_pointer)
        return graph

    def load_img(
        self, brain_id, img_path, multiscale, segmentation_path=None
    ):
        if brain_id not in self.feature_generators:
            self.feature_generators[brain_id] = FeatureGenerator(
                img_path,
                multiscale,
                anisotropy=self.ml_config.anisotropy,
                segmentation_path=segmentation_path,
                is_multimodal=self.is_multimodal,
            )

    # --- main pipeline ---
    def run(self):
        # Initialize training data
        self.set_validation_idxs()
        self.generate_proposals()
        self.generate_features()

        # Train model
        train_engine = TrainEngine(
            self.model,
            self.criterion,
            lr=self.ml_config.lr,
            n_epochs=self.ml_config.n_epochs,
        )
        self.model = train_engine.run(
            self.train_dataset_list, self.validation_dataset_list
        )
        self.save_model()

    def generate_proposals(self):
        print("brain_id - example_id - # proposals - % accepted")
        for i in range(self.n_graphs()):
            # Run
            self.pred_graphs[i].generate_proposals(
                self.graph_config.search_radius,
                complex_bool=self.graph_config.complex_bool,
                groundtruth_graph=self.gt_graphs[i],
                long_range_bool=self.graph_config.long_range_bool,
                proposals_per_leaf=self.graph_config.proposals_per_leaf,
            )

            # Report results
            brain_id = self.idx_to_ids[i]["brain_id"]
            example_id = self.idx_to_ids[i]["example_id"]
            n_proposals = self.pred_graphs[i].n_proposals()
            n_accepts = len(self.pred_graphs[i].gt_accepts)
            p_accepts = round(n_accepts / n_proposals, 4)
            print(f"{brain_id}  {example_id}  {n_proposals}  {p_accepts}")

    def generate_features(self):
        for i in range(self.n_graphs()):
            # Get proposals
            proposals_dict = {
                "proposals": self.pred_graphs[i].list_proposals(),
                "graph": self.pred_graphs[i].copy_graph()
            }

            # Generate features
            brain_id = self.idx_to_ids[i]["brain_id"]
            features = self.feature_generators[brain_id].run(
                self.pred_graphs[i],
                proposals_dict,
                self.graph_config.search_radius,
            )

            # Initialize train and validation datasets
            dataset = heterograph_datasets.init(
                self.pred_graphs[i], features,  proposals_dict["graph"]
            )
            if i in self.validation_idxs:
                self.validation_dataset_list.append(dataset)
            else:
                self.train_dataset_list.append(dataset)

    def save_model(self):
        date = datetime.today().strftime('%Y-%m-%d')
        path = os.path.join(self.output_dir, f"GNN-{date}.pth")
        torch.save(self.model, path)


class TrainEngine:
    """
    Custom class that trains graph neural networks.

    """

    def __init__(
        self,
        model,
        criterion,
        device="cuda",
        lr=1e-4,
        n_epochs=1000,
    ):
        """
        Constructs a GraphTrainer object.

        Parameters
        ----------
        model : torch.nn.Module
            Graph neural network.
        criterion : torch.nn.Module._Loss
            Loss function.
        lr : float, optional
            Learning rate. The default is 1e-4.
        n_epochs : int
            Number of epochs. The default is 1000.

        Returns
        -------
        None.

        """
        # Instance attributes
        self.criterion = criterion
        self.device = device
        self.n_epochs = n_epochs
        self.writer = SummaryWriter()

    def run(self, model, train_datasets, validation_datasets):
        """
        Trains a graph neural network in the case where "datasets" is a
        dictionary of datasets such that each corresponds to a distinct graph.

        Parameters
        ----------
        ...

        Returns
        -------
        torch.nn.Module
            Graph neural network that has been fit onto "datasets".

        """
        # Initializations
        model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)

        # Main
        best_f1 = 0
        n_upds = 0
        for epoch in range(self.n_epochs):
            # Train
            y, hat_y = [], []
            model.train()
            for dataset in train_datasets:
                # Forward pass
                hat_y_i, y_i = self.predict(model, dataset.data)
                loss = self.criterion(hat_y_i, y_i)
                self.writer.add_scalar("loss", loss, epoch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                n_upds += 1

                # Store prediction
                y.extend(ml_util.toCPU(y_i))
                hat_y.extend(ml_util.toCPU(hat_y_i))

            self.compute_metrics(y, hat_y, "train", epoch)
            self.scheduler.step()

            # Validate
            if n_upds % 1 == 0:
                y, hat_y = [], []
                model.eval()
                for dataset in validation_datasets:
                    hat_y_i, y_i = self.predict(model, dataset.data)
                    y.extend(ml_util.toCPU(y_i))
                    hat_y.extend(ml_util.toCPU(hat_y_i))
                test_score = self.compute_metrics(y, hat_y, "val", epoch)

                # Check for best
                if test_score > best_f1:
                    best_f1 = test_score
                    print("New Best F1:", best_f1)

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
        x, edge_index, edge_attr = gnn_util.get_inputs(data, self.device)
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
        hat_y = get_predictions(hat_y)

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
        return f1


# -- util --
def shuffler(my_list):
    """
    Shuffles a list of items.

    Parameters
    ----------
    my_list : list
        List to be shuffled.

    Returns
    -------
    list
        Shuffled list.

    """
    shuffle(my_list)
    return my_list


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


def get_predictions(hat_y, threshold=0.5):
    """
    Generate binary predictions based on the input probabilities.

    Parameters
    ----------
    hat_y : torch.Tensor
        Predicted probabilities generated by "self.model".
    threshold : float, optional
        The threshold value for binary classification. The default is 0.5.

    Returns
    -------
    list[int]
        Binary predictions based on the given threshold.

    """
    return (ml_util.sigmoid(np.array(hat_y)) > threshold).tolist()
