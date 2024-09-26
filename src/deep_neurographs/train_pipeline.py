"""
Created on Sat Sept 16 11:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This script trains the GraphTrace inference pipeline.

"""

from deep_neurographs.utils import img_util, util
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

    def load_img(self, path, dataset_name):
        if dataset_name not in self.imgs:
            self.imgs[dataset_name] = img_util.open_tensorstore(path, "zarr")

    def run(self):
        self.generate_proposals()

    def generate_proposals(self):
        print("dataset_name - example_id - # proposals - % accepted")
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
            dataset_name = self.idx_to_ids[i]["dataset_name"]
            example_id = self.idx_to_ids[i]["example_id"]
            n_proposals = self.pred_graphs[i].n_proposals()
            n_targets = len(self.pred_graphs[i].target_edges)
            p_accepts = round(n_targets / n_proposals, 4)
            print(f"{dataset_name}  {example_id}  {n_proposals}  {p_accepts}")

    def generate_features(self):
        # check that every example has an image that was loaded!
        pass

    def evaluate(self):
        pass
