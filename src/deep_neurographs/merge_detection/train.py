"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that detect merge mistakes.

"""

import numpy as np

from deep_neurographs.skeleton_graph import SkeletonGraph
from deep_neurographs.utils import img_util, util


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

    def load_merge_graphs(self, brain_id, segmentation_id, swc_pointer):
        key = (brain_id, segmentation_id)
        self.merge_graphs[key] = self.init_graph(swc_pointer)

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        self.img_readers[brain_id] = img_util.init_reader(img_path)
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

    # --- Get Examples ---
    def __getitem__(self, idx):
        # Extract site
        brain_id, graph, node_id, is_positive = self.get_site(idx)
        xyz = graph.node_xyz[node_id]
        voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Extract subgraph and image patches centered at site
        subgraph = self.get_subgraph(graph, node_id)
        img_patch = self.img_reader[brain_id].read(voxel, self.patch_shape)
        label_patch = self.get_label_mask(subgraph)
        patches = np.stack([img_patch, label_patch], axis=0)
        return patches, subgraph, int(is_positive)

    def get_site(self, idx):
        brain_id = self.merge_sites_df["brain_id"][idx]
        is_positive = np.random.random() > 0.5
        if is_positive:
            segmentation_id = self.merge_sites_df["segmentation_id"][idx]
            graph = self.merge_graphs[(brain_id, segmentation_id)]
        else:
            if np.random.random() > 0.5:
                return self.get_random_site()
            else:
                graph = self.gt_graphs[brain_id]
        xyz = self.merge_sites_df["xyz"][idx]
        node_id = graph.query_node(xyz)
        return brain_id, graph, node_id, is_positive

    def get_random_site(self):
        brain_id = util.sample_once(list(self.gt_graphs.keys()))
        graph = self.gt_graphs[brain_id]
        node_id = np.random.randint(0, graph.number_of_nodes())
        return brain_id, graph, node_id, False


# --- Custom Dataloader ---


# -- Helpers --
