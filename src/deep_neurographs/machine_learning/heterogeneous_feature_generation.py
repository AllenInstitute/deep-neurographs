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
def run(neurograph, search_radius, img_path, proposals=None):
    # Initializations
    features = dict()
    img_driver = "n5" if ".n5" in img_path else "zarr"
    img = img_utils.open_tensorstore(img_path, img_driver)
    proposals = neurograph.get_proposals() if proposals is None else proposals

    # Generate features
    features["nodes"] = run_on_nodes(neurograph, img)
    features["branches"] = run_on_branches(neurograph, img)
    features["proposals"] = run_on_proposals(
        neurograph, search_radius, img, proposals
    )


def run_on_nodes(neurograph, img):
    features = dict()
    features["skel"] = node_skeletal(neurograph)
    features["profiles"] = node_profiles(neurograph, img)
    return features


def run_on_branches(neurograph, img):
    return {"skel": branch_skeletal(neurograph)}


def run_on_proposals(neurograph, search_radius, img, proposals):
    """
    Generates feature vectors for every proposal in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    search_radius : float
        Search radius used to generate proposals.
    img : str
        Image stored in a GCS bucket.
    proposals : list[frozenset]
        List of proposals for which features will be generated.

    Returns
    -------
    features : dict
        Dictionary where each key-value pair corresponds to a type of feature
        vector and the numerical vector.

    """
    features = dict()
    features["skel"] = proposal_skeletal(neurograph, proposals, search_radius)
    features["profiles"] = fg.proposal_profiles(neurograph, proposals, img)
    return features


# -- Skeletal Features --
def node_skeletal(neurograph):
    features = dict()
    for i in neurograph.nodes:
        features[i] = np.concatenate(
            (
                neurograph.nodes[i]["radius"],
                neurograph.degree[i],
                len(neurograph.nodes[i]["proposals"]),
            ),
            axis=None,
        )
    return features


def branch_skeletal(neurograph):
    features = dict()
    for edge in neurograph.edges:
        features[edge] = np.concatenate(
            (
                np.mean(neurograph.edges[edge]["radius"]),
                neurograph.edge_length(edge),
            ),
            axis=None,
        )


def proposal_skeletal(neurograph, proposals, search_radius):
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
    j = list(neurograph.neighbors(i))[0]
    return get_profile_path(neurograph.orient_edge((i, j), i, key="xyz"))


def get_junction_profile_path(neurograph, i):
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


def get_node_profile_coords(neurograph, path):
    path = np.array([utils.to_voxels(xyz) for xyz in path])
    bbox = get_node_bbox(neurograph, path)
    coords = {
        "bbox": bbox,
        "path": geometry.sample_curve(path - bbox["min"], N_PROFILE_PTS),
    }
    return coords


def get_node_bbox(neurograph, coords):
    return {
        "min": np.floor(np.min(coords, axis=0)).astype(int) - 1,
        "max": np.ceil(np.max(coords, axis=0)).astype(int) + 1,
    }
