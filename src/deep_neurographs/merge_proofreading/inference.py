"""
Created on Wed August 4 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for detecting merge mistakes on skeletons generated from an automated
image segmentation.

"""

from concurrent.futures import as_completed, ProcessPoolExecutor

import networkx as nx
import numpy as np
import torch

from deep_neurographs.utils import img_util


class MergeDetector:

    def __init__(
        self,
        graph,
        img_path,
        model,
        patch_shape,
        anisotropy=(1.0, 1.0, 1.0),
        batch_size=16,
        remove_detected_sites=False,
        threshold=0.5,
        traversal_step=5,
    ):
        # Instance attributes
        self.graph = graph
        self.model = model
        self.patch_shape = patch_shape

        self.anisotropy = anisotropy
        self.batch_size = batch_size
        self.remove_detected_sites = remove_detected_sites
        self.threshold = threshold
        self.traversal_step = traversal_step  # not implemented

        # Image reader
        self.img_reader = img_util.init_reader(img_path)

    # --- Core Routines ---
    def search_graph(self):
        # Search graphs for merge mistakes
        with ProcessPoolExecutor(max_workers=16) as executor:
            #  Assign processes
            processes = list()
            visited_components = set()
            for i in self.graph.get_leafs():
                component_id = self.graph.node_component_id[i]
                if component_id not in visited_components:
                    visited_components.add(component_id)
                    processes.append(
                        executor.submit(self.search_component, i)
                    )

            # Store results
            detected_merge_sites = list()
            for process in as_completed(processes):
                detected_merge_sites.extend(process.result())

        print("# Detected Merges:", len(detected_merge_sites))

        # Optionally, remove merge mistakes from graphs
        if self.remove_detected_sites:
            pass

    def search_component(self, root):
        # Run model to detect merges
        detected_merge_sites = list()
        for node_ids, batch in self.generate_batches(root):
            # --> submit batch to available gpu
            # --> store locations with prediction above threshold
            pass

        # Process detected merges - combine nearby sites
        return detected_merge_sites

    def remove_merge_sites(self, detected_merge_sites):
        pass

    # --- Subroutines ---
    def generate_batches(self, root):
        """
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
        node_ids = list()
        patch_centers = list()
        visited = set()
        for i, j in nx.dfs_edges(self.graph, source=root):
            # Check if starting new batch
            if len(patch_centers) == 0:
                root = i
                node_ids.append(i)
                patch_centers.append(self.get_voxel(i))
                visited.add(i)

            # Check whether to yield batch
            is_node_far = self.graph.dist(root, j) > 512
            is_batch_full = len(patch_centers) >= self.batch_size
            if is_node_far or is_batch_full:
                yield self.get_batch(node_ids, patch_centers)
                node_ids = list()
                patch_centers = list()

            # Visit j
            if j not in visited:
                node_ids.append(j)
                patch_centers.append(self.get_voxel(j))
                visited.add(j)
                if len(patch_centers) == 1:
                    root = j

        # Yield any remaining nodes after the loop
        if patch_centers:
            yield self.get_batch(node_ids, patch_centers)

    def get_batch(self, node_ids, patch_centers):
        # Read superchunk
        patch_centers = np.array(patch_centers, dtype=int)
        superchunk, offset = self.read_superchunk(patch_centers)

        # Extract patches from superchunk
        batch = np.empty((len(patch_centers), 1,) + self.patch_shape)
        for i, center in enumerate(patch_centers):
            local_center = (center - offset).astype(int)
            s = img_util.get_slices(local_center, self.patch_shape)
            batch[i, 0, ...] = superchunk[s]
        return node_ids, torch.tensor(batch, dtype=torch.float)

    def read_superchunk(self, patch_centers):
        # Compute bounding box
        buffer = np.array(self.patch_shape) / 2
        start = patch_centers.min(axis=0) - buffer
        end = patch_centers.max(axis=0) + buffer + 1

        # Read image
        shape = (end - start).astype(int)
        center = (start + shape // 2).astype(int)
        superchunk = img_util.normalize(self.img_reader.read(center, shape))
        return superchunk, start

    # --- Helpers ---
    def get_voxel(self, node_id):
        xyz = self.graph.node_xyz[node_id]
        return img_util.to_voxels(xyz, self.anisotropy)
