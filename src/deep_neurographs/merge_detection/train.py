"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that detect merge mistakes.

"""

import numpy as np

from deep_neurographs.skeleton_graph import SkeletonGraph
from deep_neurographs.utils import img_util


# --- Custom Trainer ---


# --- Custom Dataset ---
class MergeDetectionGraphDataset:

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        multiscale=0,
        node_spacing=5,
        patch_shape=(84, 84, 84),
    ):
        # Instance attributes
        self.anisotropy = anisotropy
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df
        self.multiscale = multiscale
        self.patch_shape = patch_shape

        # Data structures
        self.img_readers = dict()
        self.gt_graphs = dict()
        self.gt_kdtrees = dict()
        self.merge_graphs = dict()
        self.merge_kdtrees = dict()

    # --- Load Data ---
    def init_graph(self, swc_pointer):
        graph = SkeletonGraph(
            anisotropy=self.anisotropy, node_spacing=self.node_spacing
        )
        graph.load(swc_pointer)
        return graph

    def init_kdtrees(self):
        self.gt_kdtrees = self._init_kdtree(self.gt_graphs)
        self.merge_kdtrees = self._init_kdtree(self.merge_graphs)

    def _init_kdtree(self, graphs):
        for brain_id, graph in graphs.items():
            graph.init_kdtree()

    def load_fragment_graphs(self, brain_id, segmentation_id, swc_pointer):
        key = (brain_id, segmentation_id)
        self.merge_graphs[key] = self.init_graph(swc_pointer)

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        self.img_readers[brain_id] = img_util.init_reader(img_path)
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

    # --- Get Examples ---
    def __getitem__(self, idx):
        # Get xyz coordinate of site
        brain_id = self.merge_sites_df["brain_id"][idx]
        if np.random.random() > 0.5:
            is_groundtruth = False
            xyz = self.merge_sites_df["xyz"][idx]
        else:
            is_groundtruth = True
            n = self.gt_graphs[brain_id].number_of_nodes()
            node_id = np.random.randint(0, n)
            xyz = self.gt_graphs[brain_id].node_xyz[node_id]

        # Extract patches and subgraph rooted at site
        voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)
        img_patch = self.img_reader[brain_id].read(voxel, self.patch_shape)
        subgraph = self.get_subgraph(brain_id, 
        # Annotate label patch
        label_patch = np.zeros(self.patch_shape)

    def get_subgraph(self, brain_id, node_id, is_groundtruth=True):
        pass


# --- Custom Dataloader ---


# -- Helpers --
