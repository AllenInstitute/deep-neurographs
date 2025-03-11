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
from scipy.ndimage import zoom

import numpy as np

from deep_neurographs.utils import geometry_util, img_util, util
from deep_neurographs.utils.img_util import TensorStoreReader, ZarrReader


class FeatureGenerator:
    """
    Class that generates features vectors that are used by a graph neural
    network (GNN) to classify proposals.

    """
    # Class attributes
    patch_shape = (100, 100, 100)
    n_profile_points = 16

    def __init__(
        self,
        img_path,
        multiscale,
        anisotropy=(1.0, 1.0, 1.0),
        is_multimodal=False,
        segmentation_path=None,
    ):
        """
        Initializes object that generates features for a graph.

        Parameters
        ----------
        img_path : str
            Path to the raw image assumed to be stored in a GCS bucket.
        multiscale : int
            Level in the image pyramid that voxel coordinates must index into.
        anisotropy : Tuple[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is (1.0, 1.0, 1.0).
        is_multimodal : bool, optional
            Indication of whether to generate multimodal features (i.e. image
            and label patch for each proposal). The default is False.
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
        self.multiscale = multiscale
        self.is_multimodal = is_multimodal

        # Image readers
        self.img_patch_shape = self.set_patch_shape(multiscale)
        self.img_reader = self.init_img_reader(img_path, "zarr")
        if segmentation_path is not None:
            driver = "neuroglancer_precomputed"
            self.label_patch_shape = self.set_patch_shape(0)
            self.labels_reader = self.init_img_reader(
                segmentation_path, driver
            )

    @classmethod
    def set_patch_shape(cls, multiscale):
        """
        Adjusts the patch shape by downsampling each dimension by a specified
        factor.

        Parameters
        ----------
        None

        Returns
        -------
        List[int]
            Patch shape with each dimension reduced by the multiscale.

        """
        return [s // 2 ** multiscale for s in cls.patch_shape]

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

    def init_img_reader(self, img_path, driver=None):
        """
        Initializes an image reader.

        Parameters
        ----------
        img_path : str
            Path to where the image is located.
        driver : str, optional
            Storage driver needed to read image. The default is None.

        Returns
        -------
        ImageReader
            Image reader.

        """
        if "s3" in img_path:
            return ZarrReader(img_path)
        else:
            return TensorStoreReader(img_path, driver)

    def run(self, graph, proposals_dict, radius):
        """
        Generates feature vectors for nodes, edges, and proposals in a graph.

        Parameters
        ----------
        graph : FragmentsGraph
            Graph that "proposals" belong to.
        proposals_dict : dict
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
        computation_graph = proposals_dict["graph"]
        proposals = proposals_dict["proposals"]
        if graph.leaf_kdtree is None:
            graph.init_kdtree(node_type="leaf")

        # Main
        features = {
            "nodes": self.run_on_nodes(graph, computation_graph),
            "branches": self.run_on_branches(graph, computation_graph),
            "proposals": self.run_on_proposals(graph, proposals, radius)
        }

        # Generate image patches (if applicable)
        if self.is_multimodal:
            features["patches"] = self.proposal_patches(graph, proposals)
        return features

    def run_on_nodes(self, graph, computation_graph):
        """
        Generates feature vectors for every node in "computation_graph".

        Parameters
        ----------
        graph : FragmentsGraph
            FragmentsGraph generated from a predicted segmentation.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps a node id to a feature vector.

        """
        return self.node_skeletal(graph, computation_graph)

    def run_on_branches(self, neurograph, computation_graph):
        """
        Generates feature vectors for every edge in "computation_graph".

        Parameters
        ----------
        neurograph : FragmentsGraph
            FragmentsGraph generated from a predicted segmentation.
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
            FragmentsGraph generated from a predicted segmentation.
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
    def node_skeletal(self, fragments_graph, computation_graph):
        """
        Generates skeleton-based features for nodes in "computation_graph".

        Parameters
        ----------
        fragments_graph : FragmentsGraph
            FragmentsGraph generated from a predicted segmentation.
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
                    fragments_graph.degree[i],
                    fragments_graph.nodes[i]["radius"],
                    len(fragments_graph.nodes[i]["proposals"]),
                ),
                axis=None,
            )
        return node_skeletal_features

    def branch_skeletal(self, fragments_graph, computation_graph):
        """
        Generates skeleton-based features for edges in "computation_graph".

        Parameters
        ----------
        fragments_graph : FragmentsGraph
            Fragments graph that features are to be generated from.
        computation_graph : networkx.Graph
            Graph used by GNN to classify proposals.

        Returns
        -------
        dict
            Dictionary that maps an edge id to a feature vector.

        """
        branch_skeletal_features = dict()
        for edge in fragments_graph.edges:
            branch_skeletal_features[frozenset(edge)] = np.array(
                [
                    np.mean(fragments_graph.edges[edge]["radius"]),
                    min(fragments_graph.edge_length(edge), 500) / 500,
                ],
            )
        return branch_skeletal_features

    def proposal_skeletal(self, fragments_graph, proposals, radius):
        """
        Generates skeleton-based features for "proposals".

        Parameters
        ----------
        fragments_graph : FragmentsGraph
            Graph generated from a predicted segmentation.
        proposals : List[Frozenset[int]]
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
                    fragments_graph.proposal_length(proposal) / radius,
                    fragments_graph.n_nearby_leafs(proposal, radius),
                    fragments_graph.proposal_attr(proposal, "radius"),
                    fragments_graph.proposal_directionals(proposal, 16),
                    fragments_graph.proposal_directionals(proposal, 32),
                    fragments_graph.proposal_directionals(proposal, 64),
                    fragments_graph.proposal_directionals(proposal, 128),
                ),
                axis=None,
            )
        return proposal_skeletal_features

    # --- Image features ---
    def proposal_profiles(self, fragments_graph, proposals):
        """
        Generates an image intensity profile along proposals.

        Parameters
        ----------
        fragments_graph : FragmentsGraph
            Graph that "proposals" belong to.
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
                xyz_1, xyz_2 = fragments_graph.proposal_attr(p, "xyz")
                xyz_path = geometry_util.make_line(xyz_1, xyz_2, n_points)
                threads.append(executor.submit(self.get_profile, xyz_path, p))

            # Store results
            profiles = dict()
            for thread in as_completed(threads):
                profiles.update(thread.result())
        return profiles

    def proposal_patches(self, graph, proposals):
        """
        Generates an image intensity profile along the proposal.

        Parameters
        ----------
        graph : FragmentsGraph
            Graph that "proposals" belong to.
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
            for p in proposals:
                segment_ids = graph.proposal_attr(p, "swc_id")
                proposal_xyz = graph.proposal_attr(p, "xyz")
                threads.append(
                    executor.submit(
                        self.get_patches, p, proposal_xyz, segment_ids
                    )
                )

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
        voxel_path = np.vstack([self.to_voxels(xyz) for xyz in xyz_path])
        bbox = self.get_bbox(voxel_path)
        voxel_path = geometry_util.shift_path(voxel_path, bbox["min"])
        return {"bbox": bbox, "profile_path": voxel_path}

    def get_bbox(self, voxels, is_img=True):
        center = np.round(np.mean(voxels, axis=0)).astype(int)
        shape = self.img_patch_shape if is_img else self.label_patch_shape
        bbox = {
            "min": [c - s // 2 for c, s in zip(center, shape)],
            "max": [c + s // 2 for c, s in zip(center, shape)],
        }
        return bbox

    def get_patches(self, proposal, proposal_xyz, segment_ids):
        # Image patch
        center_xyz = np.mean(proposal_xyz, axis=0)
        center = self.to_voxels(center_xyz)
        img_patch = self.img_reader.read(center, self.img_patch_shape)
        img_patch = img_util.normalize(img_patch)

        # Labels patch
        center = img_util.to_voxels(center_xyz, self.anisotropy)
        proposal_voxels = self.get_local_coordinates(center, proposal_xyz)
        label_patch = self.labels_reader.read(center, self.label_patch_shape)
        label_patch = self.relabel(label_patch, proposal_voxels, segment_ids)
        return {proposal: np.stack([img_patch, label_patch], axis=0)}

    def get_local_coordinates(self, center_voxel, xyz_pts):
        shape = self.label_patch_shape
        offset = np.array([c - s // 2 for c, s in zip(center_voxel, shape)])
        voxels = [img_util.to_voxels(xyz, self.anisotropy) for xyz in xyz_pts]
        return geometry_util.shift_path(voxels, offset)

    def relabel(self, label_patch, voxels, segment_ids):
        # Initializations
        n_points = int(geometry_util.dist(voxels[0], voxels[-1]))
        label_patch = zoom(label_patch, 1.0 / 2 ** self.multiscale, order=0)
        for i, voxel in enumerate(voxels):
            voxels[i] = [v // 2 ** self.multiscale for v in voxel]

        # Main
        relabel_patch = np.zeros(label_patch.shape)
        relabel_patch[label_patch == segment_ids[0]] = 1
        relabel_patch[label_patch == segment_ids[1]] = 1
        line = geometry_util.make_line(voxels[0], voxels[-1], n_points)
        return geometry_util.fill_path(relabel_patch, line, val=2)

    def to_voxels(self, xyz):
        return img_util.to_voxels(xyz, self.anisotropy, self.multiscale)


# --- Profile utils ---
def get_leaf_path(graph, i):
    """
    Gets path that profile will be computed over for the leaf node "i".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that node belongs to.
    i : int
        Leaf node in "graph".

    Returns
    -------
    list
        Voxel coordinates that profile is generated from.

    """
    j = graph.leaf_neighbor(i)
    xyz_path = graph.oriented_edge((i, j), i)
    return geometry_util.truncate_path(xyz_path)


def get_branching_path(graph, i):
    """
    Gets path that profile will be computed over for the branching node "i".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing node "i".
    i : int
        Branching node in "fragments_graph".

    Returns
    -------
    list
        Voxel coordinates that profile is generated from.

    """
    j1, j2 = tuple(graph.neighbors(i))
    voxels_1 = geometry_util.truncate_path(graph.oriented_edge((i, j1), i))
    voxles_2 = geometry_util.truncate_path(graph.oriented_edge((i, j2), i))
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
