"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training a model and performing inference.

Conventions:
    (1) "xyz" refers to a real world coordinate such as those from an swc file

    (2) "voxel" refers to an voxel coordinate in a whole exaspim image.

"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from random import sample

import numpy as np
import tensorstore as ts

from deep_neurographs import geometry
from deep_neurographs.utils import img_util, util

CHUNK_SIZE = [48, 48, 48]
N_BRANCH_PTS = 50
N_PROFILE_PTS = 16
N_SKEL_FEATURES = 22


class Factory:
    """
    Class that generates feature generator instances based on the specified
    model type.

    """
    @staticmethod
    def create(model_type, *args):
        if model_type == "GraphNeuralNet":
            return GraphFeatureGenerator(*args)
        else:
            return FeatureGenerator(*args)


class FeatureGenerator:
    """
    An abstract base class that generates features vectors to be classified
    and/or learned by a machine learning model.

    """

    def __init__(self, img, downsample_factor):
        """
        Instantiates a FeatureGenerator object that is used to generate
        features in a machine learning pipeline.

        Parameters
        ----------
        img : np.ndarray
            Raw image from which features will be generated.
        downsample_factor : int
            Downsampling factor that represents which level in the image
            pyramid the voxel coordinates must index into.

        Returns
        -------
        None

        """
        self.img = img
        self.downsample_factor = downsample_factor

    def run(self, neurograph, proposals_dict, radius):
        """
        Generates feature vectors for each proposal.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that "proposals" belong to.
        proposals_dict : dict
            Dictionary containing the computation graph used by gnn and
            proposals to be classified.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Feature vectors.

        """
        # Initialiations
        features = defaultdict(bool)
        proposals = proposals_dict["proposals"]
        if neurograph.leaf_kdtree is None:
            neurograph.init_kdtree(node_type="leaf")

        # Main
        features["proposals"] = {
            "skel": self.proposal_skeletal(neurograph, proposals, radius),
            "profiles": self.proposal_profiles(neurograph, proposals),
        }
        return features

    def proposal_skeletal(neurograph, proposals, radius):
        """
        Generates features from skeleton (i.e. graph) which are graph or
        geometry-based features.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that "proposals" belong to.
        proposals : list
            Proposals for which features will be generated
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Features generated from skeleton.

        """
        features = dict()
        for proposal in proposals:
            i, j = tuple(proposal)
            features[proposal] = np.concatenate(
                (
                    neurograph.proposal_length(proposal),
                    neurograph.degree[i],
                    neurograph.degree[j],
                    len(neurograph.nodes[i]["proposals"]),
                    len(neurograph.nodes[j]["proposals"]),
                    neurograph.n_nearby_leafs(proposal, radius),
                    neurograph.proposal_radii(proposal),
                    neurograph.proposal_avg_radii(proposal),
                    neurograph.proposal_directionals(proposal, 16),
                    neurograph.proposal_directionals(proposal, 32),
                    neurograph.proposal_directionals(proposal, 64),
                    neurograph.proposal_directionals(proposal, 128),
                ),
                axis=None,
            )
        return features

    def proposal_profiles(self, neurograph, proposals):
        """
        Generates an image intensity profile along the proposal.

        Parameters
        ----------
        neurograph : NeuroGraph
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
            threads = len(proposals) * [None]
            for i, proposal in enumerate(proposals):
                xyz_1, xyz_2 = neurograph.proposal_xyz(proposal)
                xyz_path = geometry.make_line(xyz_1, xyz_2, N_PROFILE_PTS)
                spec = self.get_profile_spec(xyz_path)
                threads[i] = executor.submit(
                        img_util.get_profile, self.img, spec, proposal
                    )

            # Store results
            profiles = dict()
            for thread in as_completed(threads):
                profiles.update(thread.result())
        return profiles

    def get_profile_spec(self, xyz_path):
        """
        Gets image bounding box and voxel coordinates needed to compute an
        image intensity profile.

        Parameters
        ----------
        xyz_path : numpy.ndarray
            xyz coordinates that represent an image profile path.

        Returns
        -------
        dict
            Specifications needed to compute profile for a given proposal.

        """
        voxels = self.transform_path(xyz_path)
        bbox = img_util.get_fixed_bbox(voxels, CHUNK_SIZE)
        profile_path = geometry.shift_path(voxels, bbox["min"])
        return {"bbox": bbox, "profile_path": profile_path}

    def transform_path(self, xyz_path):
        """
        Transforms "xyz_path" by converting the xyz coordinates to voxels and
        resampling "N_PROFILE_PTS" from voxel coordinates.

        Parameters
        ----------
        xyz_path : numpy.ndarray
            xyz coordinates that represent an image profile path.

        Returns
        -------
        numpy.ndarray
            Voxel coordinates that represent an image profile path.

        """
        voxels = np.zeros((len(xyz_path), 3), dtype=int)
        for i, xyz in enumerate(xyz_path):
            voxels[i] = img_util.to_voxels(xyz, self.downsample_factor)
        return voxels


class GraphFeatureGenerator(FeatureGenerator):
    """
    Class that generates features vectors that are used by a graph neural
    network to classify proposals.

    """
    def __init__(
        self, img, downsample_factor, labels=None, use_img_embedding=False
    ):
        """
        Initializes object that generates features that are used by a gnn.

        Parameters
        ----------
        img : tensorstore.Tensorstore
            Image stored in a GCS bucket.
        downsample_factor : int, optional
            Downsampling factor that accounts for which level in the image
            pyramid the voxel coordinates must index into. The default is 0.
        labels : tensorstore.TensorStore, optional
            Segmentation mask stored in a GCS bucket. The default is None.
        use_img_embedding : bool, optional
            ...

        Returns
        -------
        None

        """
        # Initialize instance attributes
        super().__init__(img, downsample_factor)
        self.labels = labels
        self.use_img_embedding = use_img_embedding

    def run(self, neurograph, proposals_dict, radius):
        """
        Generates feature vectors for a graph given a set of proposals.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that "proposals" belong to.
        proposals_dict : dict
            Dictionary that contains the items (1) "proposals" which are the
            proposals from "neurograph" that features will be generated and
            (2) "graph" which is the computation graph used by the gnn.
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
            "edge": self.run_on_edges(neurograph, computation_graph),
            "proposals": self.run_on_proposals(neurograph, proposals, radius)
        }
        return features

    def run_on_nodes(self, neurograph, computation_graph):
        """
        Generates feature vectors for every node in "computation_graph".

        Parameters
        ----------
        neurograph : NeuroGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.

        Returns
        -------
        dict
            Dictionary whose keys are feature types (i.e. skeletal) and values
            are a dictionary that maps a node id to the corresponding feature
            vector.

        """
        return {"skel": self.node_skeletal(neurograph, computation_graph)}

    def run_on_edges(self, neurograph, computation_graph):
        """
        Generates feature vectors for every edge in "computation_graph".

        Parameters
        ----------
        neurograph : NeuroGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.

        Returns
        -------
        dict
            Dictionary whose keys are feature types (i.e. skeletal) and values
            are a dictionary that maps an edge id to the corresponding feature
            vector.

        """
        return {"skel": self.edge_skeletal(neurograph, computation_graph)}

    def run_on_proposals(self, neurograph, proposals, radius):
        """
        Generates feature vectors for every proposal in "neurograph".

        Parameters
        ----------
        neurograph : NeuroGraph
            NeuroGraph generated from a predicted segmentation.
        proposals : list[frozenset]
            List of proposals for which features will be generated.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary whose keys are feature types (i.e. skeletal, profiles,
            chunks) and values are a dictionary that maps a proposal id to a
            feature vector.

        """
        # Skeleton features
        features = {
            "skel": self.proposal_skeletal(neurograph, proposals, radius)
        }

        # Image features
        if self.use_img_embedding:
            chunks, profiles = self.proposal_chunks_profiles(
                neurograph, proposals
            )
            features.update({"chunks": chunks})
            features.update({"profiles": profiles})
        else:
            profiles = self.proposal_profiles(neurograph, proposals)
            features.update({"profiles": profiles})
        return features

    # -- Skeletal Features --
    def node_skeletal(self, neurograph, computation_graph):
        """
        Generates skeleton-based features for nodes in "computation_graph".

        Parameters
        ----------
        neurograph : NeuroGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.

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

    def edge_skeletal(self, neurograph, computation_graph):
        """
        Generates skeleton-based features for edges in "computation_graph".

        Parameters
        ----------
        neurograph : NeuroGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps an edge id to a feature vector.

        """
        edge_skeletal_features = dict()
        for edge in neurograph.edges:
            edge_skeletal_features[frozenset(edge)] = np.array(
                [
                    np.mean(neurograph.edges[edge]["radius"]),
                    min(neurograph.edges[edge]["length"], 500) / 500,
                ],
            )
        return edge_skeletal_features

    def proposal_skeletal(self, neurograph, proposals, radius):
        """
        Generates skeleton-based features for "proposals".

        Parameters
        ----------
        neurograph : NeuroGraph
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
        neurograph : NeuroGraph
            NeuroGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by gnn to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to an image profile.

        """
        # Get profile specifications
        specs = dict()
        for i in computation_graph.nodes:
            if neurograph.is_leaf(i):
                profile_path = self.get_leaf_profile_path(neurograph, i)
            else:
                profile_path = self.get_branching_profile_path(neurograph, i)
            specs[i] = self.get_img_specs(profile_path)

        # Generate profiles
        with ThreadPoolExecutor() as executor:
            threads = []
            for i, spec in specs.items():
                threads.append(
                    executor.submit(img_util.get_profile, self.img, spec, i)
                )

            node_profile_features = dict()
            for thread in as_completed(threads):
                node_profile_features.update(thread.result())
        return node_profile_features

    def proposal_chunks_profiles(self, neurograph, img, proposals):
        """
        Generates an image intensity profile along each proposal.

        Parameters
        ----------
        neurograph : NeuroGraph
            Graph that "proposals" belong to.
        img : tensorstore.TensorStore
            Image stored in a GCS bucket.
        proposals : list[frozenset]
            List of proposals for which features will be generated.
        downsample_factor : int
            Downsampling factor that represents which level in the image
            pyramid the voxel coordinates must index into.

        Returns
        -------
        dict
            Dictonary such that each pair is the proposal id and profile.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = []
            for proposal in proposals:
                xyz_1, xyz_2 = neurograph.proposal_xyz(proposal)
                xyz_path = geometry.make_line(xyz_1, xyz_2, N_PROFILE_PTS)
                specs = self.get_profile_spec(xyz_path)
                threads.append(
                    executor.submit(
                        img_util.get_chunk_profile, img, specs, proposal)
                )

            # Store results
            profiles = dict()
            chunks = dict()
            for thread in as_completed(threads):
                proposal, chunk, profile = thread.result()
                chunks[proposal] = chunk
                profiles[proposal] = profile
        return chunks, profiles


# --- Profile utils ---
def get_leaf_profile_path(neurograph, i):
    """
    Gets path that profile will be computed over for the leaf node "i".

    Parameters
    ----------
    neurograph : NeuroGraph
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


def get_branching_profile_path(neurograph, i):
    """
    Gets path that profile will be computed over for the branching node "i".

    Parameters
    ----------
    neurograph : NeuroGraph
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
def get_matrix(neurographs, features, sample_ids=None):
    if sample_ids:
        return stack_feature_matrices(neurographs, features, sample_ids)
    else:
        return get_feature_matrix(neurographs, features)


def stack_feature_matrices(neurographs, features, blocks):
    # Initialize
    X = None
    y = None
    idx_transforms = {"block_to_idxs": dict(), "idx_to_edge": dict()}

    # Feature extraction
    for block_id in blocks:
        # Extract feature matrix
        idx_shift = 0 if X is None else X.shape[0]
        X_i, y_i, idx_transforms_i = get_feature_matrix(
            neurographs[block_id], features[block_id], shift=idx_shift
        )

        # Concatenate
        if X is None:
            X = deepcopy(X_i)
            y = deepcopy(y_i)
        else:
            X = np.concatenate((X, X_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)

        # Update dicts
        idx_transforms["block_to_idxs"][block_id] = idx_transforms_i[
            "block_to_idxs"
        ]
        idx_transforms["idx_to_edge"].update(idx_transforms_i["idx_to_edge"])
    return X, y, idx_transforms


def get_feature_matrix(neurograph, features, shift=0):
    # Initialize
    features = combine_features(features)
    key = sample(list(features.keys()), 1)[0]
    X = np.zeros((len(features.keys()), len(features[key])))
    y = np.zeros((len(features.keys())))
    idx_transforms = {"block_to_idxs": set(), "idx_to_edge": dict()}

    # Build
    for i, edge in enumerate(features.keys()):
        X[i, :] = features[edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
        idx_transforms["block_to_idxs"].add(i + shift)
        idx_transforms["idx_to_edge"][i + shift] = edge
    return X, y, idx_transforms


def combine_features(features):
    combined = dict()
    for edge in features["skel"].keys():
        combined[edge] = None
        for key in features.keys():
            if combined[edge] is None:
                combined[edge] = deepcopy(features[key][edge])
            else:
                combined[edge] = np.concatenate(
                    (combined[edge], features[key][edge])
                )
    return combined


# --- Utils ---
def count_features():
    """
    Counts number of features based on the "model_type".

    Parameters
    ----------
    None

    Returns
    -------
    int
        Number of features.
    """
    return N_SKEL_FEATURES + N_PROFILE_PTS + 2


def n_node_features():
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
    return {"branch": 2, "proposal": 34}


def n_edge_features():
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
    n_edge_features_dict = {
        ("proposal", "edge", "proposal"): 3,
        ("branch", "edge", "branch"): 3,
        ("branch", "edge", "proposal"): 3
    }
    return n_edge_features_dict


def get_chunk(img, labels, voxel_1, voxel_2, thread_id=None):
    # Extract chunks
    midpoint = geometry.get_midpoint(voxel_1, voxel_2).astype(int)
    if type(img) is ts.TensorStore:
        chunk = util.read_tensorstore(img, midpoint, CHUNK_SIZE)
        labels_chunk = util.read_tensorstore(labels, midpoint, CHUNK_SIZE)
    else:
        chunk = img_util.read_chunk(img, midpoint, CHUNK_SIZE)
        labels_chunk = img_util.read_chunk(labels, midpoint, CHUNK_SIZE)

    # Coordinate transform
    chunk = util.normalize(chunk)
    patch_voxel_1 = util.voxels_to_patch(voxel_1, midpoint, CHUNK_SIZE)
    patch_voxel_2 = util.voxels_to_patch(voxel_2, midpoint, CHUNK_SIZE)

    # Generate features
    path = geometry.make_line(patch_voxel_1, patch_voxel_2, N_PROFILE_PTS)
    profile = geometry.get_profile(chunk, path)
    labels_chunk[labels_chunk > 0] = 1
    labels_chunk = geometry.fill_path(labels_chunk, path, val=2)
    chunk = np.stack([chunk, labels_chunk], axis=0)

    # Output
    if thread_id:
        return thread_id, chunk, profile
    else:
        return chunk, profile
