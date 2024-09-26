"""
Created on Sat Sept 16 11:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This script trains the GraphTrace inference pipeline.

"""

from deep_neurographs.utils import util
from deep_neurographs.utils.graph_util import GraphLoader


class Trainer:
    """
    Class that is used to train a machine learning model that classifies
    proposals.

    """
    def __init__(
        self, config, model_type, output_dir=None, save_model_bool=True
    ):
        # Check for parameter errors
        if save_model_bool and not output_dir:
            raise ValueError("Must provide output_dir to save model.")

        # Set class attributes
        self.idx_to_ids = list()
        self.model_type = model_type
        self.output_dir = output_dir
        self.save_model_bool = save_model_bool

        # Set data structures for training examples
        self.gt_graphs = list()
        self.pred_graphs = list()
        self.imgs = dict()

        # Extract config settings
        self.graph_config = config.graph_config
        self.ml_config = config.ml_config
        self.graph_loader = GraphLoader(
            min_size=self.graph_config.min_size,
            progress_bar=False,
        )

    def n_examples(self):
        return len(self.gt_graphs)

    def load_example(
        self,
        gt_pointer,
        pred_pointer,
        dataset_name,
        example_id=None,
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
            {"dataset_name": dataset_name, "example_id": example_id}
        )

    def load_img(self, img_path, dataset_name):
        pass

    def run(self):
        pass

    def generate_features(self):
        # check that every example has an image that was loaded!
        pass

    def evaluate(self):
        pass
