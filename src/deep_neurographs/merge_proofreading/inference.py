"""
Created on Wed August 4 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.


---
        Generates batches by using a DFS to traverse the connected component
        containing "root". Each batch consists of (1) node ids along a path in
        the graph and (2) torch tensor of image patches centered at each node.

        Parameters
        ----------
        root : int
            Node ID that represents the starting point of the DFS.

        Returns
        -------
        generator
            Generator that yields batches to be run through a neural network.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from torch.utils.data import IterableDataset

import networkx as nx
import numpy as np
import torch

from deep_neurographs.utils import img_util, ml_util


class MergeDetector:

    def __init__(
        self,
        graph,
        img_path,
        model,
        model_path,
        patch_shape,
        anisotropy=(1.0, 1.0, 1.0),
        batch_size=16,
        prefetch=64,
        remove_detected_sites=False,
        threshold=0.5,
        traversal_step=5,
    ):
        # Instance attributes
        self.graph = graph
        self.remove_detected_sites = remove_detected_sites
        self.threshold = threshold

        # Load model
        self.model = model
        ml_util.load_model(model, model_path)

        # Initialize dataset
        self.dataset = IterableGraphDataset(
            graph,
            img_path,
            patch_shape,
            anisotropy=anisotropy,
            batch_size=batch_size,
            prefetch=prefetch,
            traversal_step=traversal_step,
        )

    # --- Core routines
    def search_merge_sites(self):
        # Iterate over dataset
        detected_merge_sites = list()
        for nodes, x_nodes in self.dataset:
            hat_y = self.predict(x_nodes)
            idxs = np.where(hat_y > self.threshold)[0]
            detected_merge_sites.extend([nodes[i] for i in idxs])
            print(hat_y)
        print("# Detected Merge Sites:", len(detected_merge_sites))

        # Optionally, remove merge mistakes from graphs
        if self.remove_detected_sites:
            pass

    def predict(self, x):
        with torch.no_grad():
            x = x.to("cuda")
            hat_y = self.model(x)
            return ml_util.to_cpu(hat_y, to_numpy=True)

    def remove_merge_sites(self, detected_merge_sites):
        pass


# --- Data Handling ---
class IterableGraphDataset(IterableDataset):

    def __init__(
        self,
        graph,
        img_path,
        patch_shape,
        anisotropy=(1.0, 1.0, 1.0),
        batch_size=16,
        prefetch=64,
        traversal_step=10,
    ):
        # Call parent class
        super().__init__()

        # Instance attributes
        self.graph = graph
        self.patch_shape = patch_shape

        self.anisotropy = anisotropy
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.traversal_step = traversal_step

        # Image reader
        self.img_reader = img_util.init_reader(img_path)

    # --- Core routines ---
    def __iter__(self):
        # Subroutines
        def submit_thread():
            try:
                nodes, patch_centers = next(batch_metadata_iter)
                thread = executor.submit(self.read_superchunk, patch_centers)
                pending[thread] = (nodes, patch_centers)
            except StopIteration:
                pass

        # Main
        batch_metadata_iter = self.generate_batch_metadata()
        with ThreadPoolExecutor(max_workers=128) as executor:
            try:
                # Prefetch batches
                pending = dict()
                for _ in range(self.prefetch):
                    submit_thread()

                # Yield batches
                while pending:
                    done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                    for thread in done:
                        # Process completed thread
                        nodes, patch_centers = pending.pop(thread)
                        img, offset = thread.result()
                        yield self.get_batch(img, offset, patch_centers, nodes)

                        # Continue submitting threads
                        submit_thread()
            finally:
                pass

    def generate_batch_metadata(self):
        """
        Generates metadata (nodes, patch_centers) used to generate batches.

        Parameters
        ----------
        None

        Returns
        -------
        iterator
            Generator that yields node IDs and patch centers used to generate
            batches across the whole graph.
        """
        visited_ids = set()
        for i in self.graph.get_leafs():
            component_id = self.graph.node_component_id[i]
            if component_id not in visited_ids:
                visited_ids.add(component_id)
                yield from self._generate_batch_metadata_for_component(i)

    def _generate_batch_metadata_for_component(self, root):
        """
        Generates metadata (nodes, patch_centers) used to generate batches
        for the connected component containing the given root node.

        Parameters
        ----------
        None

        Returns
        -------
        iterator
            Generator that yields node IDs and patch centers used to generate
            batches for the connected component containing the given root
            node.
        """
        nodes = list()
        patch_centers = list()
        visited = set()
        for i, j in nx.dfs_edges(self.graph, source=root):
            # Check if starting new batch
            if len(patch_centers) == 0:
                root = i
                last_node = i
                nodes.append(i)
                patch_centers.append(self.get_voxel(i))
                visited.add(i)

            # Check whether to yield batch
            is_node_far = self.graph.dist(root, j) > 512
            is_batch_full = len(patch_centers) == self.batch_size
            if is_node_far or is_batch_full:
                yield nodes, np.array(patch_centers, dtype=int)
                nodes = list()
                patch_centers = list()

            # Visit j
            if j not in visited:
                visited.add(j)
                is_next = self.graph.dist(last_node, j) >= self.traversal_step
                is_branching = self.graph.degree[j] >= 3
                if is_next or is_branching:
                    last_node = j
                    nodes.append(j)
                    patch_centers.append(self.get_voxel(j))
                    if len(patch_centers) == 1:
                        root = j

        # Yield any remaining nodes after the loop
        if patch_centers:
            yield nodes, np.array(patch_centers, dtype=int)

    def get_batch(self, img, offset, patch_centers, nodes):
        # Initializations
        label_mask = self.get_label_mask(nodes, img.shape, offset)
        patch_centers -= offset

        # Populate batch array
        batch = np.empty((len(patch_centers), 2,) + self.patch_shape)
        for i, center in enumerate(patch_centers):
            s = img_util.get_slices(center, self.patch_shape)
            batch[i, 0, ...] = img[s]
            batch[i, 1, ...] = label_mask[s]
        return nodes, torch.tensor(batch, dtype=torch.float)

    # --- Helpers ---
    def read_superchunk(self, patch_centers):
        # Compute bounding box
        buffer = np.array(self.patch_shape) / 2
        start = patch_centers.min(axis=0) - buffer
        end = patch_centers.max(axis=0) + buffer + 1

        # Read image
        shape = (end - start).astype(int)
        center = (start + shape // 2).astype(int)
        superchunk = img_util.normalize(self.img_reader.read(center, shape))
        return superchunk, start.astype(int)

    def get_label_mask(self, nodes, img_shape, offset):
        label_mask = np.zeros(img_shape)
        queue = list(nodes)
        visited = set(nodes)
        while queue:
            # Visit node
            node = queue.pop()
            voxel = self.get_voxel(node) - offset
            if img_util.is_contained(voxel, img_shape, buffer=3):
                label_mask[
                    max(voxel[0] - 3, 0): min(voxel[0] + 3, img_shape[0]),
                    max(voxel[1] - 3, 0): min(voxel[1] + 3, img_shape[1]),
                    max(voxel[2] - 3, 0): min(voxel[2] + 3, img_shape[2])
                ] = 1
                is_contained = True
            else:
                is_contained = False

            # Update queue
            if is_contained:
                for nb in self.graph.neighbors(node):
                    if nb not in visited:
                        queue.append(nb)
                        visited.add(nb)
        return label_mask

    def get_voxel(self, node):
        """
        Gets the voxel coordinate of the given node.

        Parameters
        ----------
        node : int
            Node ID.

        Returns
        -------
        Tuple[int]
            Voxel coordinate of the given node.
        """
        xyz = self.graph.node_xyz[node]
        return img_util.to_voxels(xyz, self.anisotropy)
