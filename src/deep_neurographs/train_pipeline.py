"""
Created on Sat Sept 16 11:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This script trains the GraphTrace inference pipeline.

"""

import os
from datetime import datetime
from random import sample

import numpy as np
from torch.nn import BCEWithLogitsLoss

from deep_neurographs.machine_learning import feature_generation
from deep_neurographs.utils import img_util, ml_util, util
from deep_neurographs.utils.graph_util import GraphLoader


class Trainer:
    """
    Class that is used to train a machine learning model that classifies
    proposals.

    """
    def __init__(
        self,
        config,
        model_type,
        criterion=None,
        output_dir=None,
        validation_ids=None,
        validation_split=0.15,
        save_model_bool=True,
    ):
        # Check for parameter errors
        if save_model_bool and not output_dir:
            raise ValueError("Must provide output_dir to save model.")

        # Set class attributes
        self.idx_to_ids = list()
        self.model_type = model_type
        self.output_dir = output_dir
        self.save_model_bool = save_model_bool
        self.validation_ids = validation_ids
        self.validation_split = validation_split

        # Set data structures for training examples
        self.gt_graphs = list()
        self.pred_graphs = list()
        self.imgs = dict()
        self.train_dataset = list()
        self.validation_dataset = list()

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
        return len(self.train_dataset)

    def n_validation_samples(self):
        return len(self.validation_dataset)

    def set_validation_idxs(self):
        if self.validation_ids is None:
            k = int(self.validation_split * self.n_examples())
            self.validation_idxs = sample(np.arange(self.n_examples), k)
        else:
            self.validation_idxs = list()
            for ids in self.validation_ids:
                for i in range(self.n_examples()):
                    same = all([ids[k] == self.idx_to_ids[i][k] for k in ids])
                    if same:
                        self.validation_idxs.append(i)

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

    def load_img(self, path, sample_id):
        if sample_id not in self.imgs:
            self.imgs[sample_id] = img_util.open_tensorstore(path, "zarr")

    # --- main pipeline ---
    def run(self):
        self.generate_proposals()
        self.generate_features()
        self.train_model()

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
            n_targets = len(self.pred_graphs[i].target_edges)
            p_accepts = round(n_targets / n_proposals, 4)
            print(f"{sample_id}  {example_id}  {n_proposals}  {p_accepts}")

    def generate_features(self):
        self.set_validation_idxs()
        for i in range(self.n_examples()):
            # Get proposals
            proposals_dict = {
                "proposals": self.pred_graphs[i].list_proposals(),
                "graph": self.pred_graphs[i].copy_graph()
            }

            # Generate features
            sample_id = self.idx_to_ids[i]["sample_id"]
            features = feature_generation.run(
                self.pred_graphs[i],
                self.imgs[sample_id],
                self.model_type,
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
            if i in self.validation_ids:
                self.validation_dataset.append(dataset)
            else:
                self.train_dataset.append(dataset)

    def train_model(self):
        pass

    def save_model(self, model):
        name = self.model_type + "-" + datetime.today().strftime('%Y-%m-%d')
        extension = ".pth" if "Net" in self.model_type else ".joblib"
        path = os.path.join(self.output_dir, name + extension)
        util.save_model(path, model)
