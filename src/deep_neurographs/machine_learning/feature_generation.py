"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training a machine learning model and performing
inference.

Conventions:
    (1) "xyz" refers to a physical coordinate such as those from an SWC file
    (2) "voxel" refers to a voxel coordinate in a whole-brain image.

Note: We assume that a segmentation mask corresponds to multiscale 0. Thus,
      the instance attribute "self.multiscale" corresponds to the multiscale
      of the input image.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from deep_neurographs.utils import geometry_util, img_util, util
from deep_neurographs.utils.img_util import TensorStoreReader, ZarrReader


class FeatureGenerator:
    """
    Class that generates features vectors that are used by a graph neural
    network (GNN) to classify proposals.

    """
    # Class attributes
    n_profile_points = 16

    def __init__(
        self,
        graph,
        img_path,
        anisotropy=(1.0, 1.0, 1.0),
        context=30,
        is_multimodal=False,
        multiscale=0,
        patch_shape=(50, 50, 50),
        segmentation_path=None,
    ):
        """
        Initializes object that generates features for a graph.

        Parameters
        ----------
        graph : FragmentsGraph
            Graph generated from a predicted segmentation which features are
            to be computed for.
        img_path : str
            Path to the raw image assumed to be stored in a GCS bucket.
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is (1.0, 1.0, 1.0).
        context : int, optional
            ...
        is_multimodal : bool, optional
            Indication of whether to generate multimodal features (i.e. image
            and label patch for each proposal). The default is False.
        multiscale : int, optional
            Level in the image pyramid that voxel coordinates must index into.
            The default is 0.
        patch_shape : Tuple[int], optional
            ...
        segmentation_path : str, optional
            Path to the segmentation assumed to be stored on a GCS bucket. The
            default is None.

        Returns
        -------
        None

        """
        # Sanity check
        if is_multimodal and not segmentation_path:
            raise ValueError("Must provide segmentation_path for multimodal!")

        # Instance attributes
        self.anisotropy = anisotropy
        self.context = context
        self.graph = graph
        self.is_multimodal = is_multimodal
        self.multiscale = multiscale if not is_multimodal else 0
        self.patch_shape = patch_shape

        # Readers
        self.img_reader = self.init_img_reader(img_path)
        if segmentation_path is not None:
            self.labels_reader = self.init_img_reader(segmentation_path)

    @classmethod
    def get_n_profile_points(cls):
        """
        Gets the number of points on an image profile.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of points on an image profile.

        """
        return cls.n_profile_points

    def init_img_reader(self, img_path):
        """
        Initializes an image reader.

        Parameters
        ----------
        img_path : str
            Path to where the image is located.

        Returns
        -------
        ImageReader
            Image reader.

        """
        if "s3" in img_path:
            return ZarrReader(img_path)
        else:
            return TensorStoreReader(img_path)

    def run(self, batch, radius):
        """
        Generates feature vectors for nodes, edges, and proposals in a graph.

        Parameters
        ----------

        batch : dict
            Dictionary that contains the items (1) "proposals" which are the
            proposals from "fragments_graph" that features will be generated
            and (2) "graph" which is the computation graph used by the GNN.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that contains different types of feature vectors for
            nodes, edges, and proposals.

        """
        # Initializations
        computation_graph = batch["graph"]
        proposals = batch["proposals"]
        if self.graph.leaf_kdtree is None:
            self.graph.init_kdtree(node_type="leaf")

        # Main
        features = {
            "nodes": self.node_skeletal(computation_graph),
            "branches": self.branch_skeletal(computation_graph),
            "proposals": self.run_on_proposals(proposals, radius)
        }

        # Generate image patches (if applicable)
        if self.is_multimodal:
            features["patches"] = self.proposal_patches(proposals)
        return features

    def run_on_proposals(self, proposals, radius):
        """
        Generates feature vectors for every proposal in "neurograph".

        Parameters
        ----------
        proposals : list[frozenset]
            List of proposals for which features will be generated.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that maps a proposal id to a feature vector.

        """
        features = self.proposal_skeletal(proposals, radius)
        if not self.is_multimodal:
            profiles = self.proposal_profiles(proposals)
            for p in proposals:
                features[p] = np.concatenate((features[p], profiles[p]))
        return features

    # -- Skeletal Features --
    def node_skeletal(self, computation_graph):
        """
        Generates skeleton-based features for nodes in "computation_graph".

        Parameters
        ----------
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to a feature vector.

        """
        skeletal_features = dict()
        for i in computation_graph.nodes:
            skeletal_features[i] = np.concatenate(
                (
                    self.graph.degree[i],
                    self.graph.nodes[i]["radius"],
                    len(self.graph.nodes[i]["proposals"]),
                ),
                axis=None,
            )
        return skeletal_features

    def branch_skeletal(self, computation_graph):
        """
        Generates skeleton-based features for edges in "computation_graph".

        Parameters
        ----------
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps an edge id to a feature vector.

        """
        skeletal_features = dict()
        for edge in computation_graph.edges:
            if edge in self.graph.edges:
                skeletal_features[frozenset(edge)] = np.array(
                    [
                        np.mean(self.graph.edges[edge]["radius"]),
                        min(self.graph.edge_length(edge), 500) / 500,
                    ],
                )
        return skeletal_features

    def proposal_skeletal(self, proposals, radius):
        """
        Generates skeleton-based features for "proposals".

        Parameters
        ----------
        proposals : List[Frozenset[int]]
            List of proposals for which features will be generated.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to a feature vector.

        """
        skeletal_features = dict()
        for proposal in proposals:
            skeletal_features[proposal] = np.concatenate(
                (
                    self.graph.proposal_length(proposal) / radius,
                    self.graph.n_nearby_leafs(proposal, radius),
                    self.graph.proposal_attr(proposal, "radius"),
                    self.graph.proposal_directionals(proposal, 16),
                    self.graph.proposal_directionals(proposal, 32),
                    self.graph.proposal_directionals(proposal, 64),
                    self.graph.proposal_directionals(proposal, 128),
                ),
                axis=None,
            )
        return skeletal_features

    # --- Image features ---
    def proposal_profiles(self, proposals):
        """
        Generates an image intensity profile along proposals.

        Parameters
        ----------
        proposals : List[Frozenset[int]]
            List of proposals for which features will be generated.

        Returns
        -------
        dict
            Dictonary such that each item is a proposal id and image
            intensity profile.

        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for p in proposals:
                n_points = self.get_n_profile_points()
                xyz_1, xyz_2 = self.graph.proposal_attr(p, "xyz")
                xyz_path = geometry_util.make_line(xyz_1, xyz_2, n_points)
                threads.append(executor.submit(self.get_profile, xyz_path, p))

            # Store results
            profiles = dict()
            for thread in as_completed(threads):
                profiles.update(thread.result())
        return profiles

    def proposal_patches(self, proposals):
        """
        Generates an image intensity profile along the proposal.

        Parameters
        ----------
        proposals : List[Frozenset[int]]
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
            for proposal in proposals:
                threads.append(executor.submit(self.get_patches, proposal))

            # Store results
            img_patches = dict()
            for thread in as_completed(threads):
                img_patches.update(thread.result())
        return img_patches

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
            Dictionary that maps an ID (e.g. node, edge, or proposal) to its
            profile.

        """
        profile = self.img_reader.read_profile(self.get_spec(xyz_path))
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
            Specifications needed to read image patch and generate profile.

        """
        # Compute bounding box
        center, shape = self.compute_bbox(xyz_path)
        bbox = {
            "min": [c - s // 2 for c, s in zip(center, shape)],
            "max": [c + s // 2 for c, s in zip(center, shape)],
        }

        # Shift voxel profile path
        voxel_path = [self.to_voxels(xyz) for xyz in xyz_path]
        voxel_path = geometry_util.shift_path(voxel_path, bbox["min"])
        voxel_path = get_inbounds(voxel_path, shape)
        return {"bbox": bbox, "profile_path": voxel_path}

    def get_patches(self, proposal):
        xyz_pts = self.graph.proposal_attr(proposal, "xyz")
        center, shape = self.compute_bbox(xyz_pts)
        img_patch = self.get_img_patch(center, shape)
        label_patch = self.get_label_patch(center, shape, proposal)
        return {proposal: np.stack([img_patch, label_patch], axis=0)}

    def get_img_patch(self, center, shape):
        img_patch = self.img_reader.read(center, shape)
        img_patch = img_util.normalize(img_patch)
        return img_util.resize(img_patch, (64, 64, 64))

    def get_label_patch(self, center, shape, proposal):
        # Read label patch
        label_patch = self.labels_reader.read(center, shape)

        # Annotate label patch
        i, j = tuple(proposal)
        label_patch = (label_patch > 0).astype(float)
        label_patch = self.annotate_edge(label_patch, center, shape, i)
        label_patch = self.annotate_edge(label_patch, center, shape, j)
        label_patch = self.annotate_proposal(
            label_patch, center, shape, proposal
        )
        return img_util.resize(label_patch, (64, 64, 64))

    def annotate_proposal(self, label_patch, center, shape, proposal):
        # Convert proposal xyz to local voxel coordinates
        proposal_xyz = self.graph.proposal_attr(proposal, "xyz")
        voxels = self.get_local_coordinates(center, shape, proposal_xyz)

        # Draw line along proposal
        n_points = int(geometry_util.dist(voxels[0], voxels[-1]))
        line = geometry_util.make_line(voxels[0], voxels[-1], n_points)
        return geometry_util.fill_path(label_patch, line, val=3)

    def annotate_edge(self, label_patch, center, shape, i):
        edge_xyz = np.vstack(self.graph.edge_attr(i, "xyz"))
        voxels = self.get_local_coordinates(center, shape, edge_xyz)
        voxels = get_inbounds(voxels, label_patch.shape)
        return geometry_util.fill_path(label_patch, voxels, val=2)

    def get_local_coordinates(self, center, shape, xyz_pts):
        offset = np.array([c - s // 2 for c, s in zip(center, shape)])
        voxels = [self.to_voxels(xyz) for xyz in xyz_pts]
        voxels = geometry_util.shift_path(voxels, offset)
        return voxels

    def compute_bbox(self, xyz_pts):
        # Compute bounds
        voxels = [self.to_voxels(xyz) for xyz in xyz_pts]
        bounds = img_util.get_minimal_bbox(voxels, self.context)

        # Transform into square
        center = np.mean(voxels, axis=0).astype(int)
        length = np.max([u - l for u, l in zip(bounds["max"], bounds["min"])])
        return center, (length, length, length)

    def to_voxels(self, xyz):
        return img_util.to_voxels(xyz, self.anisotropy, self.multiscale)


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


def init_idx_mapping(idx_to_id):
    """
    Adds dictionary item called "edge_to_index" which maps a branch/proposal
    in a FragmentsGraph to an idx that represents it's position in the feature
    matrix.

    Parameters
    ----------
    idxs : dict
        Dictionary that maps indices to edges in a FragmentsGraph.

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


# --- Helpers ---
def get_inbounds(voxels, shape):
    filtered_voxels = list()
    for voxel in voxels:
        lower_bound_bool = all(v > 0 for v in voxel)
        upper_bound_bool = all(v < s - 1 for v, s in zip(voxel, shape))
        if lower_bound_bool and upper_bound_bool:
            filtered_voxels.append(voxel)
    return filtered_voxels


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
