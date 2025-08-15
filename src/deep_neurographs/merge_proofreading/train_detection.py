"""
Created on Wed July 2 11:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Dataset and dataloader utilities for processing merge site data to train
neural networks that detect merge errors.

"""

from concurrent.futures import as_completed, ThreadPoolExecutor

import networkx as nx
import numpy as np
import random

from deep_neurographs.machine_learning.augmentation import ImageTransforms
from deep_neurographs.skeleton_graph import SkeletonGraph
from deep_neurographs.utils import img_util, ml_util, swc_util, util


class MergeSiteDataset:
    """
    Dataset class for loading and processing merge site data. The core data
    structure is the attribute "merge_sites_df" which contains metadata about
    each merge site.

    Attributes
    ----------
    merge_sites_df : pandas.DataFrame
        DataFrame containing merge sites, must contain the columns:
        "brain_id", "segmentation_id", "segment_id", and "xyz".
    anisotropy : Tuple[float], optional
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.
    context_radius : int, optional
        Radius (in microns) around merge sites used to extract rooted
        subgraph. Default is 200.
    gt_graphs : dict[str, SkeletonGraph]
        Dictionary that maps brain IDs to a graph containing ground truth
        tracings.
    gt_kdtrees : scipy.spatial.KDTree
        KD-Tree built from xyz coordinates stored in "gt_graphs".
    merge_graphs : dict[str, SkeletonGraph]
        Dictionary that maps brain IDs to a graph containing fragments that
        contain merge mistakes.
    gt_kdtrees : scipy.spatial.KDTree
        KD-Tree built from xyz coordinates stored in "merge_graphs".
    img_readers : ImageReader
        Image reader used to read raw image from cloud bucket.
    multiscale : int, optional
        Level in the image pyramid that the voxel coordinate must index into.
        Default is 0.
    node_spacing : int, optional
        Spacing between nodes in the graph. Default is 5 (microns).
    patch_shape : tuple of int, optional
        Shape of the 3D patches to extract (depth, height, width). Default is
        (96, 96, 96).
    """

    def __init__(
        self,
        merge_sites_df,
        anisotropy=(1.0, 1.0, 1.0),
        context_radius=200,
        multiscale=0,
        node_spacing=5,
        patch_shape=(96, 96, 96),
    ):
        """
        Instantiates MergeSiteDataset object.

        Parameters
        ----------
        merge_sites_df : pandas.DataFrame
            DataFrame containing merge sites, must contain the columns:
            "brain_id", "segmentation_id", "segment_id", and "xyz".
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope.
        context_radius : int, optional
            Radius (in microns) around merge sites used to extract rooted
            subgraph. Default is 200.
        multiscale : int, optional
            Level in the image pyramid that the voxel coordinate must index
            into. Default is 0.
        node_spacing : int, optional
            Spacing between nodes in the graph. Default is 5 (microns).
        patch_shape : tuple of int, optional
            Shape of the 3D patches to extract (depth, height, width). Default is
            (96, 96, 96).

        Returns
        -------
        None
        """
        # Instance attributes
        self.anisotropy = anisotropy
        self.context_radius = context_radius
        self.node_spacing = node_spacing
        self.merge_sites_df = merge_sites_df
        self.multiscale = multiscale
        self.patch_shape = patch_shape
        self.transform = ImageTransforms()

        # Data structures
        self.img_readers = dict()
        self.gt_graphs = dict()
        self.gt_kdtrees = dict()
        self.merge_graphs = dict()
        self.merge_kdtrees = dict()

    # --- Load Data ---
    def init_graph(self, swc_pointer):
        """
        Initialize a SkeletonGraph built from SWC files.

        Parameters
        ----------
        swc_pointer : any
            Pointer to SWC files to be loaded into graph.

        Returns
        -------
        SkeletonGraph
            Graph with loaded data from SWC files.
        """
        graph = SkeletonGraph(node_spacing=self.node_spacing)
        graph.load(swc_pointer)
        return graph

    def init_kdtrees(self):
        """
        Initialize KDTrees for both ground truth and merge graphs.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.gt_kdtrees = self._init_kdtree(self.gt_graphs)
        self.merge_kdtrees = self._init_kdtree(self.merge_graphs)

    def _init_kdtree(self, graphs):
        """
        Build KDTree for each graph contained in a dictionary.

        Parameters
        ----------
        graphs : dict
            Dictionary mapping brain IDs to SkeletonGraph instances.

        Returns
        -------
        dict
            Dictionary mapping brain IDs to KDTrees built from the graphs.
        """
        for brain_id, graph in graphs.items():
            graph.init_kdtree()

    def load_merge_graphs(self, brain_id, swc_pointer):
        """
        Loads and processes fragments containing merge mistakes for a given
        brain sample, then loads them into "merge_graphs" attribute.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the whole-brain dataset.
        swc_pointer : any
            Pointer to SWC files to be loaded into graph.

        Returns
        -------
        None
        """
        # Load graphs
        graph = self.init_graph(swc_pointer)

        # Filter non-merge components
        idxs = self.merge_sites_df["brain_id"] == brain_id
        merged_segment_ids = self.merge_sites_df["segment_id"][idxs].values
        for swc_id in graph.get_swc_ids():
            segment_id = swc_util.get_segment_id(swc_id)
            if str(segment_id) not in merged_segment_ids:
                component_id = util.find_key(graph.component_id_to_swc_id, swc_id)
                nodes = graph.get_nodes_with_component_id(component_id)
                graph.remove_nodes_from(nodes)
                graph.irreducible.remove_nodes_from(nodes)

        graph.relabel_nodes()
        self.merge_graphs[brain_id] = graph

    def load_gt_graphs(self, brain_id, img_path, swc_pointer):
        """
        Loads and processes ground truth tracings and image for a given brain
        sample, then loads them into "gt_graphs" and "img_readers" attributes.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the whole-brain dataset.
        swc_pointer : any
            Pointer to SWC files to be loaded into graph.

        Returns
        -------
        None
        """
        self.img_readers[brain_id] = img_util.init_reader(img_path)
        self.gt_graphs[brain_id] = self.init_graph(swc_pointer)

    # --- Get Examples ---
    def __getitem__(self, idx, use_transform):
        # Extract site
        if idx is None:
            brain_id, graph, node, is_positive = self.get_random_site()
        else:
            brain_id, graph, node, is_positive = self.get_site(idx)
        xyz = graph.node_xyz[node]
        voxel = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)

        # Extract subgraph and image patches centered at site
        subgraph = graph.get_rooted_subgraph(node, self.context_radius)
        img_patch = self.get_img_patch(brain_id, voxel)
        label_patch = self.get_label_mask(subgraph)

        # Stack image channels
        try:
            patches = np.stack([img_patch, label_patch], axis=0)
        except ValueError:
            img_patch = img_util.pad_to_shape(img_patch, self.patch_shape)
            patches = np.stack([img_patch, label_patch], axis=0)
            print("Stack Channels Failed -", brain_id, voxel, img_patch.shape)

        # Apply image augmentation (if applicable)
        if use_transform:
            _, patches = self.transform(idx, patches)
        return patches, subgraph, int(is_positive)

    def get_site(self, idx):
        # Extract graph
        brain_id = self.merge_sites_df["brain_id"].iloc[idx]
        is_positive = idx > 0
        if is_positive:
            graph = self.merge_graphs[brain_id]
        else:
            graph = self.gt_graphs[brain_id]

        # Extract site info
        xyz = self.merge_sites_df["xyz"].iloc[abs(idx)]
        node = graph.query_node(xyz)
        return brain_id, graph, node, is_positive

    def get_random_site(self):
        # Sample graph
        brain_id = util.sample_once(list(self.gt_graphs.keys()))
        graph = self.gt_graphs[brain_id]

        # Sample node on graph
        if random.random() < 0.5:
            node = np.random.randint(0, graph.number_of_nodes())
        else:
            node = util.sample_once(graph.get_branchings())
        return brain_id, graph, node, False

    def get_img_patch(self, brain_id, center):
        img_patch = self.img_readers[brain_id].read(center, self.patch_shape)
        return img_util.normalize(img_patch)

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
        return 2 * len(self.merge_sites_df)

    def count_fragments(self):
        cnt = 0
        for graph in self.merge_graphs.values():
            cnt += nx.number_connected_components(graph)
        return cnt


class MergeSiteDataloader:
    """
    DataLoader that uses multithreading to read image patches from the cloud
    to form batches.

    Attributes
    ----------
    batch_size : int
        Number of examples in each batch.
    dataset : MergeSiteDataset
        Dataset containing merge sites to be loaded during training.
    idxs : List[int]
        Example indexes to load from "dataset". Note: "dataset" contains both
        train and validation data.
    use_random_sites : bool
        Indication of whether to use random non-merge sites sampled from
        ground tracings during training.
    use_transform : bool
        Indication of whether to use data augmentation during training.
    """

    def __init__(
        self,
        dataset,
        idxs,
        batch_size=32,
        use_random_sites=False,
        use_transform=False
    ):
        """
        Instantiates MergeSiteDataloader object.

        Parameters
        ----------
        dataset : MergeSiteDataset
            Dataset containing merge sites to be loaded during training.
        idxs : List[int]
            Example indexes to load from "dataset". Note: "dataset" contains
            both train and validation data.
        batch_size : int, optional
            Number of examples in each batch. Default is 32.
        use_random_sites : bool, optional
            Indication of whether to use random non-merge sites sampled from
            ground tracings during training. Default is False.
        use_transform : bool, optional
            Indication of whether to use data augmentation during training.
            Default is False.

        Returns
        -------
        None
        """
        # Instance attributes
        self.batch_size = batch_size
        self.dataset = dataset
        self.idxs = self._load_idxs(idxs)
        self.use_random_sites = use_random_sites
        self.use_transform = use_transform

    def __iter__(self):
        """
        Generates batches of examples used during training and validation.

        Parameters
        ----------
        None

        Returns
        -------
        iterator
            Generates batch of examples used during training and validation.
        """
        random.shuffle(self.idxs)
        for i_start in range(0, len(self.idxs), self.batch_size):
            yield self._load_batch(i_start)

    def _load_batch(self, i_start):
        # Generate batch
        batch_size = min(self.batch_size, len(self.idxs) - i_start)
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for i_offset in range(batch_size):
                # Get idx
                idx = self.idxs[i_start + i_offset]
                if idx < 0 and self.use_random_sites:
                    idx = None if np.random.random() > 0.5 else idx

                # Submit job
                threads.append(
                    executor.submit(
                        self.dataset.__getitem__, idx, self.use_transform
                    )
                )

            # Process results
            patch_shape = (batch_size, 2,) + self.dataset.patch_shape
            patches = np.empty(patch_shape, dtype=np.float32)
            labels = np.empty((batch_size, 1), dtype=np.float32)
            for i, thread in enumerate(as_completed(threads)):
                patch, _, label = thread.result()
                patches[i] = patch
                labels[i] = label
        return ml_util.to_tensor(patches), ml_util.to_tensor(labels)

    def _load_idxs(self, idxs):
        negative_idxs = -np.arange(0, len(idxs))
        idxs.extend(negative_idxs.tolist())
        return idxs


# -- Helpers --
def shift_voxel(voxel, center, patch_shape):
    voxel = [v - c + s // 2 for v, c, s in zip(voxel, center, patch_shape)]
    return tuple(voxel)
