"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that classify proposals.

"""

import os
from copy import deepcopy
from datetime import datetime
from random import sample, shuffle

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from deep_neurographs.machine_learning.feature_generation import FeatureGenerator
from deep_neurographs.utils import gnn_util, img_util, ml_util, util
from deep_neurographs.utils.gnn_util import toCPU
from deep_neurographs.utils.graph_util import GraphLoader

LR = 1e-3
N_EPOCHS = 500
SCHEDULER_GAMMA = 0.7
SCHEDULER_STEP_SIZE = 100
WEIGHT_DECAY = 1e-3


class TrainPipeline:
    """
    Class that is used to train a machine learning model that classifies
    proposals.

    """
    def __init__(
        self,
        config,
        model,
        model_type,
        criterion=None,
        output_dir=None,
        use_img_embedding=False,
        validation_ids=None,
        save_model_bool=True,
    ):
        # Check for parameter errors
        if save_model_bool and not output_dir:
            raise ValueError("Must provide output_dir to save model.")

        # Set class attributes
        self.feature_generators = dict()
        self.idx_to_ids = list()
        self.model = model
        self.model_type = model_type
        self.output_dir = output_dir
        self.save_model_bool = save_model_bool
        self.use_img_embedding = use_img_embedding
        self.validation_ids = validation_ids

        # Set data structures for training examples
        self.gt_graphs = list()
        self.pred_graphs = list()
        self.train_dataset_list = list()
        self.validation_dataset_list = list()

        # Train parameters
        self.criterion = criterion if criterion else BCEWithLogitsLoss()
        self.validation_ids = validation_ids

        # Extract config settings
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config
        self.graph_loader = GraphLoader(
            min_size=self.graph_config.min_size,
            progress_bar=False,
        )

    # --- getters/setters ---
    def n_examples(self):
        return len(self.gt_graphs)

    def n_train_examples(self):
        return len(self.train_dataset_list)

    def n_validation_samples(self):
        return len(self.validation_dataset_list)

    def set_validation_idxs(self):
        if self.validation_ids is None:
            k = int(self.ml_config.validation_split * self.n_examples())
            self.validation_idxs = sample(np.arange(self.n_examples), k)
        else:
            self.validation_idxs = list()
            for ids in self.validation_ids:
                for i in range(self.n_examples()):
                    same = all([ids[k] == self.idx_to_ids[i][k] for k in ids])
                    if same:
                        self.validation_idxs.append(i)
        assert len(self.validation_idxs) > 0, "No validation data!"

    # --- loaders ---
    def load_example(
        self,
        gt_pointer,
        pred_pointer,
        sample_id,
        example_id=None,
        pred_id=None,
        metadata_path=None,
    ):
        # Read metadata
        if metadata_path:
            origin, shape = util.read_metadata(metadata_path)
        else:
            origin, shape = None, None

        # Load graphs
        self.gt_graphs.append(self.graph_loader.run(gt_pointer))
        self.pred_graphs.append(
            self.graph_loader.run(
                pred_pointer,
                img_patch_origin=origin,
                img_patch_shape=shape,
            )
        )

        # Set example ids
        self.idx_to_ids.append(
            {
                "sample_id": sample_id,
                "example_id": example_id,
                "pred_id": pred_id,
            }
        )

    def load_img(
        self, sample_id, img_path, downsample_factor, label_path=None
    ):
        if sample_id not in self.feature_generators:
            self.feature_generators[sample_id] = FeatureGenerator(
                img_path,
                downsample_factor,
                label_path=label_path,
                use_img_embedding=self.use_img_embedding,
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

        # Save model (if applicable)
        if self.save_model_bool:
            self.save_model()

    def generate_proposals(self):
        print("sample_id - example_id - # proposals - % accepted")
        for i in range(self.n_examples()):
            # Run
            self.pred_graphs[i].generate_proposals(
                self.graph_config.search_radius,
                complex_bool=self.graph_config.complex_bool,
                groundtruth_graph=self.gt_graphs[i],
                long_range_bool=self.graph_config.long_range_bool,
                progress_bar=False,
                proposals_per_leaf=self.graph_config.proposals_per_leaf,
                trim_endpoints_bool=self.graph_config.trim_endpoints_bool,
            )

            # Report results
            sample_id = self.idx_to_ids[i]["sample_id"]
            example_id = self.idx_to_ids[i]["example_id"]
            n_proposals = self.pred_graphs[i].n_proposals()
            n_accepts = len(self.pred_graphs[i].gt_accepts)
            p_accepts = round(n_accepts / n_proposals, 4)
            print(f"{sample_id}  {example_id}  {n_proposals}  {p_accepts}")

    def generate_features(self):
        for i in range(self.n_examples()):
            # Get proposals
            proposals_dict = {
                "proposals": self.pred_graphs[i].list_proposals(),
                "graph": self.pred_graphs[i].copy_graph()
            }

            # Generate features
            sample_id = self.idx_to_ids[i]["sample_id"]
            features = self.feature_generators[sample_id].run(
                self.pred_graphs[i],
                proposals_dict,
                self.graph_config.search_radius,
            )

            # Initialize train and validation datasets
            dataset = ml_util.init_dataset(
                self.pred_graphs[i],
                features,
                self.model_type,
                computation_graph=proposals_dict["graph"]
            )
            if i in self.validation_idxs:
                self.validation_dataset_list.append(dataset)
            else:
                self.train_dataset_list.append(dataset)

    def save_model(self):
        name = self.model_type + "-" + datetime.today().strftime('%Y-%m-%d')
        extension = ".pth" if "Net" in self.model_type else ".joblib"
        path = os.path.join(self.output_dir, name + extension)
        ml_util.save_model(path, self.model, self.model_type)


class TrainEngine:
    """
    Custom class that trains graph neural networks.

    """

    def __init__(
        self,
        model,
        criterion,
        lr=LR,
        n_epochs=N_EPOCHS,
        weight_decay=WEIGHT_DECAY,
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
            Learning rate. The default is the global variable LR.
        n_epochs : int
            Number of epochs. The default is the global variable N_EPOCHS.
        weight_decay : float
            Weight decay used in optimizer. The default is the global variable
            WEIGHT_DECAY.

        Returns
        -------
        None.

        """
        # Training
        self.model = model  # .to("cuda:0")
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.init_scheduler()
        self.writer = SummaryWriter()

    def init_scheduler(self):
        self.scheduler = StepLR(
            self.optimizer,
            step_size=SCHEDULER_STEP_SIZE,
            gamma=SCHEDULER_GAMMA,
        )

    def run(self, train_dataset_list, validation_dataset_list):
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
        best_score = -np.inf
        best_ckpt = None
        for epoch in range(self.n_epochs):
            # Train
            y, hat_y = [], []
            self.model.train()
            for dataset in train_dataset_list:
                # Forward pass
                hat_y_i, y_i = self.predict(dataset.data)
                loss = self.criterion(hat_y_i, y_i)
                self.writer.add_scalar("loss", loss, epoch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Store prediction
                y.extend(toCPU(y_i))
                hat_y.extend(toCPU(hat_y_i))

            self.compute_metrics(y, hat_y, "train", epoch)
            self.scheduler.step()

            # Validate
            if epoch % 10 == 0:
                y, hat_y = [], []
                self.model.eval()
                for dataset in validation_dataset_list:
                    hat_y_i, y_i = self.predict(dataset.data)
                    y.extend(toCPU(y_i))
                    hat_y.extend(toCPU(hat_y_i))
                test_score = self.compute_metrics(y, hat_y, "val", epoch)

                # Check for best
                if test_score > best_score:
                    best_score = test_score
                    best_ckpt = deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_ckpt)
        return self.model

    def predict(self, data):
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
        x, edge_index, edge_attr = gnn_util.get_inputs(data, "HeteroGNN")
        hat_y = self.model(x, edge_index, edge_attr)
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


def fit_random_forest(model, dataset):
    model.fit(dataset.data.x, dataset.data.y)
    return model


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