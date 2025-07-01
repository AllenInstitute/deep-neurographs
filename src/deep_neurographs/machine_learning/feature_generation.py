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
        context=40,
        is_multimodal=False,
        multiscale=0,
        patch_shape=(64, 64, 64),
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
            print("Segmentation path not provided for inference!")

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
        else:
            self.labels_reader = None

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
        if self.graph.leaf_kdtree is None:
            self.graph.init_kdtree(node_type="leaf")

        # Main
        features = {
            "nodes": self.node_skeletal(batch["graph"]),
            "branches": self.branch_skeletal(batch["graph"]),
        }
        features.update(self.run_on_proposals(batch["proposals"], radius))
        return features

    def run_on_proposals(self, proposals, radius):
        """
        Generates feature vectors for every proposal in graph.

        Parameters
        ----------
        proposals : List[Frozenset[int]
            Proposals for which features will be generated.
        radius : float
            Search radius used to generate proposals.

        Returns
        -------
        dict
            Dictionary that maps a proposal id to a feature vector.
        """
        # Generate features
        skel_features = self.proposal_skeletal(proposals, radius)
        if self.is_multimodal:
            patches, profiles = self.proposal_patches(proposals)
        else:
            profiles = self.proposal_profiles(proposals)

        # Concatenate image profiles
        for p in proposals:
            skel_features[p] = np.concatenate((skel_features[p], profiles[p]))

        # Output
        features = {"proposals": skel_features}
        if self.is_multimodal:
            features["patches"] = patches
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
                        min(self.graph.edge_length(edge), 2000) / 1000,
                    ],
                )
        return skeletal_features

    def proposal_skeletal(self, proposals, radius):
        """
        Generates skeleton-based features for "proposals".

        Parameters
        ----------
        proposals : List[Frozenset[int]]
            Proposals for which features will be generated.
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
            Proposals for which features will be generated.

        Returns
        -------
        dict
            Dictonary such that each item is a proposal id and image
            intensity profile.
        """
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for proposal in proposals:
                threads.append(executor.submit(self.get_profile, proposal))

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
            patches, profiles = dict(), dict()
            for thread in as_completed(threads):
                proposal, patches_i, profile_i = thread.result()
                patches[proposal] = patches_i
                profiles[proposal] = profile_i
        return patches, profiles

    def get_profile(self, proposal):
        """
        Gets the image intensity profile given xyz coordinates that form a
        path.

        Parameters
        ----------
        proposal : Frozenset[int]
            Identifier of profile.

        Returns
        -------
        dict
            Dictionary that maps an ID (e.g. node, edge, or proposal) to its
            profile.
        """
        # Compute bounding box
        xyz_pts = self.graph.proposal_attr(proposal, "xyz")
        center, shape = self.compute_bbox(xyz_pts)

        # Read image patch
        patch = self.img_reader.read(center, shape)
        intensity = np.percentile(patch, 99.9)
        patch = np.minimum(patch / intensity, 1)

        # Get image profile
        profile = list()
        for voxel in self.get_profile_line(center, shape, proposal):
            try:
                profile.append(patch[tuple(voxel)])
            except:
                profile.append(0)
        profile.extend([np.mean(profile), np.std(profile)])
        return {proposal: profile}

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
        center, patch_shape = self.compute_bbox(xyz_path)
        shape = self.img_reader.shape()
        shape = shape[2:] if len(shape) == 5 else shape
        iterator = list(zip(center, patch_shape, shape))
        bbox = {
            "min": [max(0, c - ps // 2) for c, ps, _ in iterator],
            "max": [min(c + ps // 2, s - 1) for c, ps, s in iterator],
        }

        # Shift voxel profile path
        voxel_path = [self.to_voxels(xyz) for xyz in xyz_path]
        voxel_path = geometry_util.shift_path(voxel_path, bbox["min"])
        voxel_path = get_inbounds(voxel_path, shape)
        return {"bbox": bbox, "profile_path": voxel_path}

    def get_patches(self, proposal):
        # Compute bounding box
        xyz_pts = self.graph.proposal_attr(proposal, "xyz")
        center, shape = self.compute_bbox(xyz_pts)

        # Compute profile coordinates
        # to do...

        # Get patches
        img_patch, profile = self.get_img_patch(center, shape, proposal)
        label_patch = self.get_label_patch(center, shape, proposal)
        patches = np.stack([img_patch, label_patch], axis=0)
        return proposal, patches, profile

    def get_img_patch(self, center, shape, proposal):
        # Get image patch
        try:
            patch = self.img_reader.read(center, shape)
            intensity = np.percentile(patch, 99.9)
            patch = np.minimum(patch / intensity, 1)
        except Exception as e:
            print("Except:", e)
            print("center:", center)
            print("shape:", shape)
            print("img.shape:", shape)
            patch = np.zeros(shape)

        # Get image profile
        profile_path = self.get_profile_line(center, shape, proposal)
        profile = [patch[tuple(voxel)] for voxel in profile_path]
        profile.extend([np.mean(profile), np.std(profile)])
        return img_util.resize(patch, self.patch_shape), profile

    def get_label_patch(self, center, shape, proposal):
        # Read label patch
        if self.labels_reader:
            patch = self.labels_reader.read(center, shape)
        else:
            patch = np.zeros(shape)

        # Annotate label patch
        i, j = tuple(proposal)
        patch = (patch > 0).astype(float)
        patch = self.annotate_edge(patch, center, shape, i)
        patch = self.annotate_edge(patch, center, shape, j)
        patch = self.annotate_proposal(patch, center, shape, proposal)
        return img_util.resize(patch, self.patch_shape)

    def annotate_proposal(self, patch, center, shape, proposal):
        profile_path = self.get_profile_line(center, shape, proposal, False)
        return img_util.fill_path(patch, profile_path, val=3)

    def annotate_edge(self, patch, center, shape, i):
        edge_xyz = np.vstack(self.graph.edge_attr(i, "xyz"))
        voxels = self.get_local_coordinates(center, shape, edge_xyz)
        voxels = get_inbounds(voxels, patch.shape)
        return img_util.fill_path(patch, voxels, val=2)

    # --- Helpers ---
    def compute_bbox(self, xyz_pts):
        # Compute bounds
        voxels = [self.to_voxels(xyz) for xyz in xyz_pts]
        bounds = img_util.get_minimal_bbox(voxels, self.context)

        # Transform into square
        center = np.mean(voxels, axis=0).astype(int)
        length = np.max([u - l for u, l in zip(bounds["max"], bounds["min"])])
        return center, (length, length, length)

    def get_profile_line(self, center, shape, proposal, fixed_len=True):
        # Convert proposal xyz to local voxel coordinates
        proposal_xyz = self.graph.proposal_attr(proposal, "xyz")
        voxels = self.get_local_coordinates(center, shape, proposal_xyz)

        # Draw line along proposal
        n_pts = 16 if fixed_len else geometry_util.dist(voxels[0], voxels[-1])
        return geometry_util.make_line(voxels[0], voxels[-1], int(n_pts))

    def get_local_coordinates(self, center, shape, xyz_pts):
        offset = np.array([c - s // 2 for c, s in zip(center, shape)])
        voxels = [self.to_voxels(xyz) for xyz in xyz_pts]
        voxels = geometry_util.shift_path(voxels, offset)
        return voxels

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


def get_node_dict():
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
