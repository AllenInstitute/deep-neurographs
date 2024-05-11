"""
Created on Sat May 9 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training and performing inference with a heterogenous
graph neural network.

"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from deep_neurographs import geometry, img_utils, utils
from deep_neurographs.machine_learning import feature_generation as fg

WINDOW = [5, 5, 5]
N_PROFILE_PTS = 10
NODE_PROFILE_DEPTH = 15


# -- Wrappers --
def run(neurograph, img_path, search_radius, proposals=None):
    """
    Generates features for proposals, edges, and nodes.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    img_path : str
        Path to an image on a GCS bucket.
    search_radius : float
        Search radius used to generate proposals.
    proposals : list[frozenset], optional
        List of proposals for which features will be generated. The default
        is None.

    Returns
    -------
    dict
        Dictionary that contains different types of feature vectors for
        proposals, edges, and nodes.

    """
    # Initializations
    features = dict()
    img_driver = "n5" if ".n5" in img_path else "zarr"
    img = img_utils.open_tensorstore(img_path, img_driver)
    proposals = neurograph.get_proposals() if proposals is None else proposals

    # Generate features
    features["nodes"] = run_on_nodes(neurograph, img)
    features["branches"] = run_on_branches(neurograph, img)
    features["proposals"] = run_on_proposals(
        neurograph, img, proposals, search_radius
    )
    return features


def run_on_nodes(neurograph, img):
    """
    Generates feature vectors for every node in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    img : str
        Image stored in a GCS bucket.

    Returns
    -------
    dict
        Dictionary that maps nodes to a dictionary of different types of
        feature vector.

    """
    features = dict()
    features["skel"] = node_skeletal(neurograph)
    features["profiles"] = node_profiles(neurograph, img)
    return features


def run_on_branches(neurograph, img):
    """
    Generates feature vectors for every edge in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    img : str
        Image stored in a GCS bucket.

    Returns
    -------
    dict
        Dictionary that maps edges to a dictionary of different types of
        feature vector.

    """
    return {"skel": branch_skeletal(neurograph)}


def run_on_proposals(neurograph, img, proposals, search_radius):
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
    search_radius : float
        Search radius used to generate proposals.

    Returns
    -------
    dict
        Dictionary that maps proposals to a dictionary of different types of
        feature vector.

    """
    features = dict()
    features["skel"] = proposal_skeletal(neurograph, proposals, search_radius)
    features["profiles"] = fg.proposal_profiles(neurograph, proposals, img)
    return features


# -- Skeletal Features --
def node_skeletal(neurograph):
    """
    Generates skeleton-based features for nodes in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.

    Returns
    -------
    dict
        Dictionary that maps nodes to the corresponding feature vectors.

    """
    features = dict()
    for i in neurograph.nodes:
        features[i] = np.concatenate(
            (
                neurograph.degree[i],
                neurograph.nodes[i]["radius"],
                len(neurograph.nodes[i]["proposals"]),
            ),
            axis=None,
        )
    return features


def branch_skeletal(neurograph):
    """
    Generates skeleton-based features for edges in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.

    Returns
    -------
    dict
        Dictionary that maps edges to the corresponding feature vectors.

    """
    features = dict()
    for edge in neurograph.edges:
        features[frozenset(edge)] = np.concatenate(
            (
                np.mean(neurograph.edges[edge]["radius"]),
                neurograph.edge_length(edge),
            ),
            axis=None,
        )
    return features


def proposal_skeletal(neurograph, proposals, search_radius):
    """
    Generates skeleton-based features for "proposals".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    proposals : list[frozenset]
        List of proposals for which features will be generated.
    search_radius : float
        Search radius used to generate proposals.

    Returns
    -------
    dict
        Dictionary that maps nodes to the corresponding feature vectors.

    """
    features = dict()
    for proposal in proposals:
        i, j = tuple(proposal)
        features[proposal] = np.concatenate(
            (
                neurograph.proposal_length(proposal),
                fg.n_nearby_leafs(neurograph, proposal, search_radius),
                fg.get_radii(neurograph, proposal),
                fg.get_directionals(neurograph, proposal, 8),
                fg.get_directionals(neurograph, proposal, 16),
                fg.get_directionals(neurograph, proposal, 32),
                fg.get_directionals(neurograph, proposal, 64),
            ),
            axis=None,
        )
    return features


# -- Image features --
def node_profiles(neurograph, img):
    """
    Generates proposals for nodes in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    img : str
        Image stored in a GCS bucket.

    Returns
    -------
    dict
        Dictionary that maps nodes to image profiles.

    """
    # Generate coordinates
    coords = dict()
    for i in neurograph.nodes:
        if neurograph.degree[i] == 1:
            path = get_leaf_profile_path(neurograph, i)
        else:
            path = get_junction_profile_path(neurograph, i)
        coords[i] = get_node_profile_coords(neurograph, path)

    # Generate profiles
    with ThreadPoolExecutor() as executor:
        threads = []
        for i, coords_i in coords.items():
            threads.append(executor.submit(fg.get_profile, img, coords_i, i))

    # Process results
    profiles = dict()
    for thread in as_completed(threads):
        i, profile = thread.result()
        profiles[i] = profile
    return profiles


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
        xyz coordinates that profile will be computed over.

    """
    j = list(neurograph.neighbors(i))[0]
    return get_profile_path(neurograph.orient_edge((i, j), i, key="xyz"))


def get_junction_profile_path(neurograph, i):
    """
    Gets path that profile will be computed over for the junction node "i".

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a predicted segmentation.
    i : int
        Junction node in "neurograph".

    Returns
    -------
    list
        xyz coordinates that profile will be computed over.

    """
    # Get branches
    nbs = list(neurograph.neighbors(i))
    xyz_list_1 = neurograph.orient_edge((i, nbs[0]), i, key="xyz")
    xyz_list_2 = neurograph.orient_edge((i, nbs[1]), i, key="xyz")

    # Get profile paths
    path_1 = get_profile_path(xyz_list_1)
    path_2 = get_profile_path(xyz_list_2)
    return np.vstack([np.flip(path_1, axis=0), path_2])


def get_profile_path(xyz_list):
    path_length = 0
    for i in range(1, len(xyz_list)):
        if i > 0:
            path_length += geometry.dist(xyz_list[i - 1], xyz_list[i])
        if path_length >= NODE_PROFILE_DEPTH:
            break
    return xyz_list[0:i, :]


def get_node_profile_coords(neurograph, xyz_arr):
    xyz_arr = np.array([utils.to_voxels(xyz) for xyz in xyz_arr])
    bbox = get_bbox(neurograph, xyz_arr)
    coords = {
        "bbox": bbox,
        "path": geometry.sample_curve(xyz_arr - bbox["min"], N_PROFILE_PTS),
    }
    return coords


def get_bbox(neurograph, xyz_arr):
    return {
        "min": np.floor(np.min(xyz_arr, axis=0)).astype(int) - 1,
        "max": np.ceil(np.max(xyz_arr, axis=0)).astype(int) + 1,
    }
