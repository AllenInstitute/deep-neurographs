"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that detect merge mistakes.

"""

from deep_neurographs.skeleton_graph import SkeletonGraph
from deep_neurographs.utils import graph_util as gutil


# --- Custom Trainer ---


# --- Custom Dataset ---
class MergeDetectionGraphDataset:

    def __init__(self, merge_sites_df, anisotropy=(1.0, 1.0, 1.0), node_spacing=5):
        # Instance attributes
        self.anisotropy = anisotropy
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df

        # Data structures
        self.imgs = dict()
        self.gt_graphs = dict()
        self.merge_graphs = dict()

    def init_graph(self, swc_pointer):
        graph = SkeletonGraph(
            anisotropy=self.anisotropy, node_spacing=self.node_spacing
        )
        graph.load(swc_pointer)
        return graph

    def load_fragment_graphs(self, brain_id, segmentation_id, swc_pointer):
        key = (brain_id, segmentation_id)
        self.merge_graphs[key] = self.init_graph(swc_pointer)

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        self.imgs[brain_id] = img_path
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

# --- Custom Dataloader ---


# -- Helpers --
