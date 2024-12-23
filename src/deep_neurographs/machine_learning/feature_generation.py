"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training a machine learning model and performing
inference.

Conventions:
    (1) "xyz" refers to a real world coordinate such as those from an swc file
    (2) "voxel" refers to an voxel coordinate in a whole exaspim image.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
from scipy.ndimage import zoom

from deep_neurographs import geometry
from deep_neurographs.utils import img_util, util


class FeatureGenerator:
    """
    Class that generates features vectors that are used by a graph neural
    network (GNN) to classify proposals.

    """
    # Class attributes
    patch_shape = [96, 96, 96]
    n_profile_points = 16

    def __init__(
        self,
        img_path,
        multiscale,
        anisotropy=[1.0, 1.0, 1.0],
        label_path=None,
        is_multimodal=False,
    ):
        """
        Initializes object that generates features for a graph.

        Parameters
        ----------
        img_path : str
            Path to the raw image assumed to be stored in a GCS bucket.
        multiscale : int
            Level in the image pyramid that voxel coordinates must index into.
        anisotropy : ArrayLike, optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        label_path : str, optional
            Path to the segmentation assumed to be stored on a GCS bucket. The
            default is None.
        is_multimodal : bool, optional
            Indication of whether to generate multimodal features (i.e. image
            and label patch for each proposal). The default is False.

        Returns
        -------
        None

        """
        # General instance attributes
        self.anisotropy = anisotropy
        self.multiscale = multiscale
        self.is_multimodal = is_multimodal

        # Open images
        driver = "n5" if ".n5" in img_path else "zarr"
        self.img = img_util.open_tensorstore(img_path, driver=driver)
        if label_path:
            self.labels = img_util.open_tensorstore(label_path)
        else:
            self.labels = None

        # Set chunk shapes
        self.img_patch_shape = self.set_patch_shape(multiscale)
        self.label_patch_shape = self.set_patch_shape(0)

        # Validate embedding requirements
        if self.is_multimodal and not label_path:
            raise("Must provide labels to generate image embeddings")

    @classmethod
    def set_patch_shape(cls, multiscale):
        """
        Adjusts the chunk shape by downsampling each dimension by a specified
        factor.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Adjusted chunk shape with each dimension reduced by the downsample
            factor.

        """
        return [s // 2 ** multiscale for s in cls.patch_shape]

    @classmethod
    def get_n_profile_points(cls):
        return cls.n_profile_points

    def run(self, neurograph, proposals_dict, radius):
        """
        Generates feature vectors for nodes, edges, and
        proposals in a graph.

        Parameters
        ----------
        neurograph : FragmentsGraph
            Graph that "proposals" belong to.
        proposals_dict : dict
            Dictionary that contains the items (1) "proposals" which are the
            proposals from "neurograph" that features will be generated and
            (2) "graph" which is the computation graph used by the GNN.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that contains different types of feature vectors for
            nodes, edges, and proposals.

        """
        # Initializations
        computation_graph = proposals_dict["graph"]
        proposals = proposals_dict["proposals"]
        if neurograph.leaf_kdtree is None:
            neurograph.init_kdtree(node_type="leaf")

        # Main
        features = {
            "nodes": self.run_on_nodes(neurograph, computation_graph),
            "branches": self.run_on_branches(neurograph, computation_graph),
            "proposals": self.run_on_proposals(neurograph, proposals, radius)
        }

        # Generate image patches (if applicable)
        if self.is_multimodal:
            features["patches"] = self.proposal_patches(neurograph, proposals)
        return features

    def run_on_nodes(self, neurograph, computation_graph):
        """
        Generates feature vectors for every node in "computation_graph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to a feature vector.

        """
        return self.node_skeletal(neurograph, computation_graph)

    def run_on_branches(self, neurograph, computation_graph):
        """
        Generates feature vectors for every edge in "computation_graph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps an branch id to a feature vector.

        """
        return self.branch_skeletal(neurograph, computation_graph)

    def run_on_proposals(self, neurograph, proposals, radius):
        """
        Generates feature vectors for every proposal in "neurograph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            NeuroGraph generated from a predicted segmentation.
        proposals : list[frozenset]
            List of proposals for which features will be generated.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that maps a proposal id to a feature vector.

        """
        features = self.proposal_skeletal(neurograph, proposals, radius)
        if not self.is_multimodal:
            profiles = self.proposal_profiles(neurograph, proposals)
            for p in proposals:
                features[p] = np.concatenate((features[p], profiles[p]))
        return features

    # -- Skeletal Features --
    def node_skeletal(self, neurograph, computation_graph):
        """
        Generates skeleton-based features for nodes in "computation_graph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to a feature vector.

        """
        node_skeletal_features = dict()
        for i in computation_graph.nodes:
            node_skeletal_features[i] = np.concatenate(
                (
                    neurograph.degree[i],
                    neurograph.nodes[i]["radius"],
                    len(neurograph.nodes[i]["proposals"]),
                ),
                axis=None,
            )
        return node_skeletal_features

    def branch_skeletal(self, neurograph, computation_graph):
        """
        Generates skeleton-based features for edges in "computation_graph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            Fragments graph that features are to be generated from.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps an edge id to a feature vector.

        """
        branch_skeletal_features = dict()
        for edge in neurograph.edges:
            branch_skeletal_features[frozenset(edge)] = np.array(
                [
                    np.mean(neurograph.edges[edge]["radius"]),
                    min(neurograph.edges[edge]["length"], 500) / 500,
                ],
            )
        return branch_skeletal_features

    def proposal_skeletal(self, neurograph, proposals, radius):
        """
        Generates skeleton-based features for "proposals".

        Parameters
        ----------
        neurograph : FragmentsGraph
            NeuroGraph generated from a predicted segmentation.
        proposals : list[frozenset]
            List of proposals for which features will be generated.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to a feature vector.

        """
        proposal_skeletal_features = dict()
        for proposal in proposals:
            proposal_skeletal_features[proposal] = np.concatenate(
                (
                    neurograph.proposal_length(proposal) / radius,
                    neurograph.n_nearby_leafs(proposal, radius),
                    neurograph.proposal_radii(proposal),
                    neurograph.proposal_directionals(proposal, 16),
                    neurograph.proposal_directionals(proposal, 32),
                    neurograph.proposal_directionals(proposal, 64),
                    neurograph.proposal_directionals(proposal, 128),
                ),
                axis=None,
            )
        return proposal_skeletal_features

    # --- Image features ---
    def node_profiles(self, neurograph, computation_graph):
        """
        Generates image profiles for nodes in "computation_graph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to an image profile.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = computation_graph.number_of_nodes() * [None]
            for idx, i in enumerate(computation_graph.nodes):
                # Get profile path
                if neurograph.is_leaf(i):
                    xyz_path = self.get_leaf_path(neurograph, i)
                else:
                    xyz_path = self.get_branching_path(neurograph, i)

                # Assign
                threads[idx] = executor.submit(
                    img_util.get_profile, self.img, self.get_spec(xyz_path), i
                )

            # Store results
            node_profile_features = dict()
            for thread in as_completed(threads):
                node_profile_features.update(thread.result())
        return node_profile_features

    def proposal_profiles(self, neurograph, proposals):
        """
        Generates an image intensity profile along the proposal.

        Parameters
        ----------
        neurograph : FragmentsGraph
            Graph that "proposals" belong to.
        proposals : list[frozenset]
            List of proposals for which features will be generated.

        Returns
        -------
        dict
            Dictonary such that each pair is the proposal id and image
            intensity profile.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for p in proposals:
                n_points = self.get_n_profile_points()
                xyz_1, xyz_2 = neurograph.proposal_xyz(p)
                xyz_path = geometry.make_line(xyz_1, xyz_2, n_points)
                threads.append(executor.submit(self.get_profile, xyz_path, p))

            # Store results
            profiles = dict()
            for thread in as_completed(threads):
                profiles.update(thread.result())
        return profiles

    def proposal_patches(self, neurograph, proposals):
        """
        Generates an image intensity profile along the proposal.

        Parameters
        ----------
        neurograph : FragmentsGraph
            Graph that "proposals" belong to.
        proposals : list[frozenset]
            List of proposals for which features will be generated.

        Returns
        -------
        dict
            Dictonary such that each pair is the proposal id and image
            intensity profile.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for p in proposals:
                labels = neurograph.proposal_labels(p)
                xyz_path = np.vstack(neurograph.proposal_xyz(p))
                threads.append(
                    executor.submit(self.get_patch, labels, xyz_path, p)
                )

            # Store results
            chunks = dict()
            for thread in as_completed(threads):
                chunks.update(thread.result())
        return chunks

    def get_profile(self, xyz_path, profile_id):
        """
        Gets the image intensity profile given xyz coordinates that form a
        path.

        Parameters
        ----------
        xyz_path : numpy.ndarray
            xyz coordinates of a profile path.
        profile_id : hashable
            Identifier of profile.

        Returns
        -------
        dict
            Dictionary that maps an id (e.g. node, edge, or proposal) to its
            profile.

        """
        profile = img_util.read_profile(self.img, self.get_spec(xyz_path))
        profile.extend(list(util.get_avg_std(profile)))
        return {profile_id: profile}

    def get_spec(self, xyz_path):
        """
        Gets image bounding box and voxel coordinates needed to compute an
        image intensity profile or extract image patch.

        Parameters
        ----------
        xyz_path : numpy.ndarray
            xyz coordinates of a profile path.

        Returns
        -------
        dict
            Specifications needed to compute a profile.

        """
        voxels = self.transform_path(xyz_path)
        bbox = self.get_bbox(voxels)
        profile_path = geometry.shift_path(voxels, bbox["min"])
        return {"bbox": bbox, "profile_path": profile_path}

    def transform_path(self, xyz_path):
        """
        Converts "xyz_path" from world to voxel coordinates.

        Parameters
        ----------
        xyz_path : numpy.ndarray
            xyz coordinates of a profile path.

        Returns
        -------
        numpy.ndarray
            Voxel coordinates of given path.

        """
        voxels = np.zeros((len(xyz_path), 3), dtype=int)
        for i, xyz in enumerate(xyz_path):
            voxels[i] = img_util.to_voxels(xyz, self.anisotropy, self.multiscale)
        return voxels

    def get_bbox(self, voxels, is_img=True):
        center = np.round(np.mean(voxels, axis=0)).astype(int)
        shape = self.img_patch_shape if is_img else self.label_patch_shape
        bbox = {
            "min": [c - s // 2 for c, s in zip(center, shape)],
            "max": [c + s // 2 for c, s in zip(center, shape)],
        }
        return bbox

    def get_patch(self, labels, xyz_path, proposal):
        # Initializations
        center = np.mean(xyz_path, axis=0)
        voxels = [img_util.to_voxels(xyz, self.anisotropy) for xyz in xyz_path]

        # Read patches
        img_patch = self.read_img_patch(center)
        label_patch = self.read_label_patch(voxels, labels)
        return {proposal: np.stack([img_patch, label_patch], axis=0)}

    def read_img_patch(self, xyz_centroid):
        center = img_util.to_voxels(xyz_centroid, self.anisotropy, self.multiscale)
        img_patch = img_util.read_tensorstore(
            self.img, center, self.img_patch_shape
        )
        return img_util.normalize(img_patch)

    def read_label_patch(self, voxels, labels):
        bbox = self.get_bbox(voxels, is_img=False)
        label_patch = img_util.read_tensorstore_with_bbox(self.labels, bbox)
        voxels = geometry.shift_path(voxels, bbox["min"])
        return self.relabel(label_patch, voxels, labels)

    def relabel(self, label_patch, voxels, labels):
        # Initializations
        n_points = self.get_n_profile_points()
        scaling_factor = 2 ** self.multiscale
        label_patch = zoom(label_patch, 1.0 / scaling_factor, order=0)
        for i, voxel in enumerate(voxels):
            voxels[i] = [v // scaling_factor for v in voxel]

        # Main
        relabel_patch = np.zeros(label_patch.shape)
        relabel_patch[label_patch == labels[0]] = 1
        relabel_patch[label_patch == labels[1]] = 2
        line = geometry.make_line(voxels[0], voxels[-1], n_points)
        return geometry.fill_path(relabel_patch, line, val=-1)


# --- Profile utils ---
def get_leaf_path(neurograph, i):
    """
    Gets path that profile will be computed over for the leaf node "i".

    Parameters
    ----------
    neurograph : FragmentsGraph
        NeuroGraph generated from a predicted segmentation.
    i : int
        Leaf node in "neurograph".

    Returns
    -------
    list
        Voxel coordinates that profile is generated from.

    """
    j = neurograph.leaf_neighbor(i)
    xyz_path = neurograph.oriented_edge((i, j), i)
    return geometry.truncate_path(xyz_path)


def get_branching_path(neurograph, i):
    """
    Gets path that profile will be computed over for the branching node "i".

    Parameters
    ----------
    neurograph : FragmentsGraph
        NeuroGraph generated from a predicted segmentation.
    i : int
        branching node in "neurograph".

    Returns
    -------
    list
        Voxel coordinates that profile is generated from.

    """
    j_1, j_2 = tuple(neurograph.neighbors(i))
    voxels_1 = geometry.truncate_path(neurograph.oriented_edge((i, j_1), i))
    voxles_2 = geometry.truncate_path(neurograph.oriented_edge((i, j_2), i))
    return np.vstack([np.flip(voxels_1, axis=0), voxles_2])


# --- Build feature matrix ---
def get_matrix(features, gt_accepts=set()):
    # Initialize matrices
    key = util.sample_once(list(features.keys()))
    x = np.zeros((len(features.keys()), len(features[key])))
    y = np.zeros((len(features.keys())))

    # Populate
    idx_to_id = dict()
    for i, id_i in enumerate(features):
        idx_to_id[i] = id_i
        x[i, :] = features[id_i]
        y[i] = 1 if id_i in gt_accepts else 0
    return x, y, init_idx_mapping(idx_to_id)


def get_patches_matrix(patches, id_to_idx):
    patch = util.sample_once(list(patches.values()))
    x = np.zeros((len(id_to_idx),) + patch.shape)
    for key, patch in patches.items():
        x[id_to_idx[key], ...] = patch
    return x


def stack_matrices(neurographs, features, blocks):
    x, y = None, None
    for block_id in blocks:
        x_i, y_i, _ = get_matrix(features[block_id])
        if x is None:
            x = deepcopy(x_i)
            y = deepcopy(y_i)
        else:
            x = np.concatenate((x, x_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)
    return x, y


def init_idx_mapping(idx_to_id):
    """
    Adds dictionary item called "edge_to_index" which maps a branch/proposal
    in a neurograph to an idx that represents it's position in the feature
    matrix.

    Parameters
    ----------
    idxs : dict
        Dictionary that maps indices to edges in some neurograph.

    Returns
    -------
    dict
        Updated dictionary.

    """
    idx_mapping = {
        "idx_to_id": idx_to_id,
        "id_to_idx": {v: k for k, v in idx_to_id.items()}
    }
    return idx_mapping


# --- Utils ---
def get_node_dict(is_multimodal=False):
    """
    Returns the number of features for different node types.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary containing the number of features for each node type

    """
    if is_multimodal:
        return {"branch": 2, "proposal": 16}
    else:
        return {"branch": 2, "proposal": 34}


def get_edge_dict():
    """
    Returns the number of features for different edge types.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary containing the number of features for each edge type

    """
    edge_dict = {
        ("proposal", "edge", "proposal"): 3,
        ("branch", "edge", "branch"): 3,
        ("branch", "edge", "proposal"): 3
    }
    return edge_dict
