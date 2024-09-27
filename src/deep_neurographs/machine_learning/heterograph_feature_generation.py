"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training and performing inference with a heterogenous
graph neural network.

"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from deep_neurographs import geometry
from deep_neurographs.machine_learning import feature_generation as feats
from deep_neurographs.utils import img_util

N_PROFILE_PTS = 16
NODE_PROFILE_DEPTH = 16
WINDOW = [5, 5, 5]


def generate_hgnn_features(
    neurograph, img, proposals_dict, radius, downsample_factor
):
    """
    Generates features for a heterogeneous graph neural network model.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    img : str
        Image stored on a GCS bucket.
    proposals_dict : dict
        Dictionary containing the computation graph used by gnn and proposals
        to be classified.
    radius : float
        Search radius used to generate proposals.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Dictionary that contains different types of feature vectors for
        nodes, edges, and proposals.

    """
    computation_graph = proposals_dict["graph"]
    proposals = proposals_dict["proposals"]
    features = {
        "nodes": run_on_nodes(
            neurograph, computation_graph, img, downsample_factor
        ),
        "edges": run_on_edges(neurograph, computation_graph),
        "proposals": run_on_proposals(
            neurograph, img, proposals, radius, downsample_factor
        ),
    }
    return features


def run_on_nodes(neurograph, computation_graph, img, downsample_factor):
    """
    Generates feature vectors for every node in "computation_graph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    computation_graph : networkx.Graph
        Graph used by gnn to classify proposals.
    img : str
        Image stored in a GCS bucket.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Dictionary whose keys are feature types (i.e. skeletal) and values are
        a dictionary that maps a node id to the corresponding feature vector.

    """
    return {"skel": node_skeletal(neurograph, computation_graph)}


def run_on_edges(neurograph, computation_graph):
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
        Dictionary whose keys are feature types (i.e. skeletal) and values are
        a dictionary that maps an edge id to the corresponding feature vector.

    """
    return {"skel": edge_skeletal(neurograph, computation_graph)}


def run_on_proposals(neurograph, img, proposals, radius, downsample_factor):
    """
    Generates feature vectors for every proposal in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    img : str
        Image stored in a GCS bucket.
    proposals : list[frozenset]
        List of proposals for which features will be generated.
    radius : float
        Search radius used to generate proposals.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Dictionary whose keys are feature types (i.e. skeletal and profiles)
        and values are a dictionary that maps a proposal id to the
        corresponding feature vector.

    """
    proposal_features = {
        "skel": proposal_skeletal(neurograph, proposals, radius),
        "profiles": feats.proposal_profiles(
            neurograph, img, proposals, downsample_factor
        ),
    }
    return proposal_features


# -- Skeletal Features --
def node_skeletal(neurograph, computation_graph):
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
        Dictionary that maps a node id to the corresponding feature vector.

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


def edge_skeletal(neurograph, computation_graph):
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
        Dictionary that maps an edge id to the corresponding feature vector.

    """
    edge_skeletal_features = dict()
    for edge in neurograph.edges:
        edge_skeletal_features[frozenset(edge)] = np.concatenate(
            (
                np.mean(neurograph.edges[edge]["radius"]),
                neurograph.edge_length(edge) / 1000,
            ),
            axis=None,
        )
    return edge_skeletal_features


def proposal_skeletal(neurograph, proposals, radius):
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
        Dictionary that maps a node id to the corresponding feature vector.

    """
    proposal_skeletal_features = dict()
    for proposal in proposals:
        i, j = tuple(proposal)
        proposal_skeletal_features[proposal] = np.concatenate(
            (
                neurograph.proposal_length(proposal),
                neurograph.n_nearby_leafs(proposal, radius),
                neurograph.proposal_radii(proposal),
                neurograph.proposal_directionals(proposal, 8),
                neurograph.proposal_directionals(proposal, 16),
                neurograph.proposal_directionals(proposal, 32),
                neurograph.proposal_directionals(proposal, 64),
            ),
            axis=None,
        )
    return proposal_skeletal_features


# -- image features --
def node_profiles(neurograph, computation_graph, img, downsample_factor):
    """
    Generates image profiles for nodes in "computation_graph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    computation_graph : networkx.Graph
        Graph used by gnn to classify proposals.
    img : str
        Image to be read from.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Dictionary that maps a node id to the corresponding image profile.

    """
    # Get specifications to compute profiles
    specs = dict()
    for i in computation_graph.nodes:
        if neurograph.degree[i] == 1:
            profile_path = get_leaf_profile_path(neurograph, i)
        else:
            profile_path = get_branching_profile_path(neurograph, i)
        specs[i] = get_node_profile_specs(profile_path, downsample_factor)

    # Generate profiles
    with ThreadPoolExecutor() as executor:
        threads = []
        for i, specs_i in specs.items():
            threads.append(executor.submit(feats.get_profile, img, specs_i, i))

        node_profile_features = dict()
        for thread in as_completed(threads):
            node_profile_features.update(thread.result())
    return node_profile_features


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
    return get_profile_path(neurograph.oriented_edge((i, j), i, key="xyz"))


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
    nbs = list(neurograph.neighbors(i))
    voxels_1 = get_profile_path(neurograph.oriented_edge((i, nbs[0]), i))
    voxles_2 = get_profile_path(neurograph.oriented_edge((i, nbs[1]), i))
    return np.vstack([np.flip(voxels_1, axis=0), voxles_2])


def get_profile_path(xyz_path):
    """
    Gets a sub-path from "xyz_path" that has a path length of at most
    "NODE_PROFILE_DEPTH" microns.

    Parameters
    ----------
    xyz_path : numpy.ndarray
        xyz coordinates that correspond to some edge in a neurograph from
        which the profile path is extracted from.

    Returns
    -------
    numpy.ndarray
        xyz coordinates that an image profile will be generated from.

    """
    # Check for degeneracy
    if xyz_path.shape[0] == 1:
        xyz_path = np.vstack([xyz_path, xyz_path - 0.01])

    # Truncate path
    length = 0
    for i in range(1, xyz_path.shape[0]):
        length += geometry.dist(xyz_path[i - 1], xyz_path[i])
        if length >= NODE_PROFILE_DEPTH:
            break
    return xyz_path[0:i, :]


def get_node_profile_specs(xyz_path, downsample_factor):
    """
    Gets image bounding box and voxel coordinates needed to compute an image
    profile.

    Parameters
    ----------
    xyz_path : numpy.ndarray
        xyz coordinates that represent an image profile path.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Specifications needed to compute image profile for a given proposal.

    """
    voxels = transform_path(xyz_path, downsample_factor)
    bbox = img_util.get_minimal_bbox(voxels, buffer=1)
    return {"bbox": bbox, "profile_path": shift_path(voxels, bbox)}


def transform_path(xyz_path, downsample_factor):
    """
    Transforms "xyz_path" by converting the xyz coordinates to voxels and
    resampling "N_PROFILE_PTS" from voxel coordinates.

    Parameters
    ----------
    xyz_path : numpy.ndarray
        xyz coordinates that represent an image profile path.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    numpy.ndarray
        Voxel coordinates that represent an image profile path.

    """
    # Main
    voxels = list()
    for xyz in xyz_path:
        voxels.append(
            img_util.to_voxels(xyz, downsample_factor=downsample_factor)
        )

    # Finish
    voxels = np.array(voxels)
    if voxels.shape[0] < 5:
        voxels = check_degenerate(voxels)
    return geometry.sample_curve(voxels, N_PROFILE_PTS)


def shift_path(voxels, bbox):
    """
    Shifts "voxels" by subtracting the min coordinate in "bbox".

    Parameters
    ----------
    voxels : numpy.ndarray
        Voxel coordinates to be shifted.
    bbox : dict
        Coordinates of a bounding box that contains "voxels".

    Returns
    -------
    numpy.ndarray
        Voxels shifted by min coordinate in "bbox".

    """
    return [voxel - bbox["min"] for voxel in voxels]


def check_degenerate(voxels):
    """
    Checks whether "voxels" contains at least two unique points. If False, the
    unique voxel coordinate is perturbed and added to "voxels".

    Parameters
    ----------
    voxels : numpy.ndarray
        Voxel coordinates to be checked.

    Returns
    -------
    numpy.ndarray
        Voxel coordinates that form a non-degenerate path.

    """
    if np.unique(voxels, axis=0).shape[0] == 1:
        voxels = np.vstack(
            [voxels, voxels[0, :] + np.array([1, 1, 1], dtype=int)]
        )
    return voxels


def n_node_features():
    return {'branch': 2, 'proposal': 34}


def n_edge_features():
    n_edge_features_dict = {
        ('proposal', 'edge', 'proposal'): 3,
        ('branch', 'edge', 'branch'): 3,
        ('branch', 'edge', 'proposal'): 3
    }
    return n_edge_features_dict
