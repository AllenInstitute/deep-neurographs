"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training machine learning models that detect merge mistakes.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
import torch

from deep_neurographs.skeleton_graph import SkeletonGraph
from deep_neurographs.utils import img_util, swc_util, util


# --- Trainer ---
class MergeDetectionTrainer:
    pass


# --- Dataset ---
class MergeDetectionGraphDataset:

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        context_radius=200,
        multiscale=0,
        node_spacing=5,
        patch_shape=(84, 84, 84),
    ):
        # Instance attributes
        self.anisotropy = anisotropy
        self.context_radius = context_radius
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
        graph = SkeletonGraph(node_spacing=self.node_spacing)
        graph.load(swc_pointer)
        return graph

    def init_kdtrees(self):
        self.gt_kdtrees = self._init_kdtree(self.gt_graphs)
        self.merge_kdtrees = self._init_kdtree(self.merge_graphs)

    def _init_kdtree(self, graphs):
        for brain_id, graph in graphs.items():
            graph.init_kdtree()

    def load_merge_graphs(self, brain_id, swc_pointer):
        # Load graphs
        graph = self.init_graph(swc_pointer)

        # Filter non-merge components
        idxs = self.merge_sites_df["brain_id"] == brain_id
        merged_segment_ids = self.merge_sites_df["segment_id"][idxs].values
        for swc_id in graph.get_swc_ids():
            segment_id = swc_util.get_segment_id(swc_id)
            if str(segment_id) not in merged_segment_ids:
                component_id = find_key(graph.component_id_to_swc_id, swc_id)
                nodes = graph.get_nodes_with_component_id(component_id)
                graph.remove_nodes_from(nodes)
                graph.irreducible.remove_nodes_from(nodes)

        graph.relabel_nodes()
        self.merge_graphs[brain_id] = graph

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        self.img_readers[brain_id] = img_util.init_reader(img_path)
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

    # --- Get Examples ---
    def __getitem__(self, idx):
        # Extract site
        brain_id, graph, node, is_positive = self.get_site(idx)
        xyz = graph.node_xyz[node]
        voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Extract subgraph and image patches centered at site
        subgraph = graph.get_rooted_subgraph(node, self.context_radius)
        img_patch = self.img_readers[brain_id].read(voxel, self.patch_shape)
        label_patch = self.get_label_mask(subgraph)
        patches = np.stack([img_patch, label_patch], axis=0)
        return patches, subgraph, int(is_positive)

    def get_site(self, idx):
        brain_id = self.merge_sites_df["brain_id"][idx]
        is_positive = np.random.random() > 0.5
        if is_positive:
            graph = self.merge_graphs[brain_id]
        else:
            if np.random.random() > 0.5:
                return self.get_random_site()
            else:
                graph = self.gt_graphs[brain_id]
        xyz = self.merge_sites_df["xyz"][idx]
        node = graph.query_node(xyz)
        return brain_id, graph, node, is_positive

    def get_random_site(self):
        brain_id = util.sample_once(list(self.gt_graphs.keys()))
        graph = self.gt_graphs[brain_id]
        node = np.random.randint(0, graph.number_of_nodes())
        return brain_id, graph, node, False

    def get_label_mask(self, subgraph):
        # Initializations
        label_mask = np.zeros(self.patch_shape)
        xyz = subgraph.node_xyz[0]
        center = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Populate label mask
        for i in subgraph.nodes:
            xyz = subgraph.node_xyz[i]
            voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)
            voxel = shift_voxel(voxel, center, self.patch_shape)
            if img_util.is_contained(voxel, self.patch_shape, buffer=3):
                i, j, k = voxel
                label_mask[i-3:i+3, j-3:j+3, k-3:k+3] = 1
        return label_mask

    # --- Helpers ---
    def __len__(self):
        return len(self.merge_sites_df)


# --- Dataloader ---
class MergeDetectionGraphDataloader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.
    """

    def __init__(self, dataset, batch_size=32):
        # Instance attributes
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        for idx in range(0, len(self.dataset), self.batch_size):
            yield self._load_batch(idx)

    def _load_batch(self, start_idx):
        # Compute batch size
        n_remaining_examples = len(self.dataset) - start_idx
        batch_size = min(self.batch_size, n_remaining_examples)

        # Generate batch
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for idx_shift in range(batch_size):
                idx = start_idx + idx_shift
                threads.append(executor.submit(self.dataset.__getitem__, idx))

            # Process results
            patches = np.zeros((batch_size, 2,) + self.dataset.patch_shape)
            labels = np.zeros((batch_size))
            for i, thread in enumerate(as_completed(threads)):
                patch, _, label = thread.result()
                patches[i] = patch
                labels[i] = label
        return to_tensor(patches), to_tensor(labels)


# -- Helpers --
def find_key(my_dict, target_value):
    for key, value in my_dict.items():
        if value == target_value:
            return key
    return None


def shift_voxel(voxel, center, patch_shape):
    voxel = [v - c + s // 2 for v, c, s in zip(voxel, center, patch_shape)]
    return tuple(voxel)


def to_tensor(arr):
    """
    Converts a numpy array to a tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to tensor.

    """
    return torch.tensor(arr, dtype=torch.float)
