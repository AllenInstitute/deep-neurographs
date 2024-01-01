"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training and inference.

"""

from copy import deepcopy
from random import sample

import numpy as np

from deep_neurographs import geometry_utils, utils

CHUNK_SIZE = [64, 64, 64]
WINDOW = [5, 5, 5]
N_PROFILE_POINTS = 10
N_SKEL_FEATURES = 11
SUPPORTED_MODELS = [
    "AdaBoost",
    "RandomForest",
    "FeedForwardNet",
    "ConvNet",
    "MultiModalNet",
]


# -- Wrappers --
def generate_mutable_features(
    neurograph,
    model_type,
    anisotropy=[1.0, 1.0, 1.0],
    img_path=None,
    labels_path=None,
):
    """
    Generates feature vectors for every edge proposal in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    model_type : str
        Indication of model to be trained. Options include: AdaBoost,
        RandomForest, FeedForwardNet, ConvNet, MultiModalNet.
    anisotropy : list[float], optional
        Real-world to image coordinates scaling factor for (x, y, z).
        The default is [1.0, 1.0, 1.0].
    img_path : str, optional
        Path to raw image. The default is None.
    labels_path : str, optional
        Path to predicted segmentation. The default is None.

    Returns
    -------
    features : dict
        Dictionary where each key-value pair corresponds to a type of feature
        vector and the numerical vector.

    """
    features = {"skel": generate_mutable_skel_features(neurograph)}
    if model_type in ["ConvNet", "MultiModalNet"]:
        features["img_chunks"] = generate_img_chunks(
            neurograph, img_path, labels_path, anisotropy=anisotropy
        )
    if model_type != "ConvNet":
        features["img_profile"] = generate_img_profiles(
            neurograph, img_path, anisotropy=anisotropy
        )
    return features


# -- Edge feature extraction --
def generate_img_chunks(
    neurograph, img_path, labels_path, anisotropy=[1.0, 1.0, 1.0]
):
    """
    Generates an image chunk for each edge proposal such that the centroid of
    the image chunk is the midpoint of the edge proposal. Image chunks contain
    two channels: raw image and predicted segmentation.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    img_path : str
        Path to raw image.
    labels_path : str
        Path to predicted segmentation.
    anisotropy : list[float], optional
        Real-world to image coordinates scaling factor for (x, y, z).
        The default is [1.0, 1.0, 1.0].

    Returns
    -------
    features : dict
        Dictonary such that each pair is the edge id and image chunk.

    """
    features = dict()
    origin = utils.apply_anisotropy(neurograph.origin, return_int=True)
    img, labels = utils.get_superchunks(
        img_path, labels_path, origin, neurograph.shape, from_center=False
    )
    img = utils.normalize_img(img)
    for edge in neurograph.mutable_edges:
        # Compute image coordinates
        i, j = tuple(edge)
        xyz_i = utils.world_to_img(neurograph, i)
        xyz_j = utils.world_to_img(neurograph, j)

        # Extract chunks
        midpoint = geometry_utils.get_midpoint(xyz_i, xyz_j).astype(int)
        img_chunk = utils.get_chunk(img, midpoint, CHUNK_SIZE)
        labels_chunk = utils.get_chunk(labels, midpoint, CHUNK_SIZE)

        # Mark path
        if neurograph.optimize_alignment or neurograph.optimize_path:
            xyz_list = neurograph.to_patch_coords(edge, midpoint, CHUNK_SIZE)
            path = geometry_utils.sample_path(xyz_list, N_PROFILE_POINTS)
        else:
            d = int(geometry_utils.dist(xyz_i, xyz_j) + 5)
            img_coords_i = utils.img_to_patch(xyz_i, midpoint, CHUNK_SIZE)
            img_coords_j = utils.img_to_patch(xyz_j, midpoint, CHUNK_SIZE)
            path = geometry_utils.make_line(img_coords_i, img_coords_j, d)

        labels_chunk[labels_chunk > 0] = 1
        labels_chunk = geometry_utils.fill_path(labels_chunk, path)
        features[edge] = np.stack([img_chunk, labels_chunk], axis=0)

    return features


def generate_img_profiles(neurograph, path, anisotropy=[1.0, 1.0, 1.0]):
    """
    Generates an image intensity profile along each edge proposal.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    path : str
        Path to raw image.
    anisotropy : list[float], optional
        Real-world to image coordinates scaling factor for (x, y, z).
        The default is [1.0, 1.0, 1.0].

    Returns
    -------
    features : dict
        Dictonary such that each pair is the edge id and image intensity
        profile.    

    """
    features = dict()
    origin = utils.apply_anisotropy(neurograph.origin, return_int=True)
    img = utils.get_superchunk(
        path, "zarr", origin, neurograph.shape, from_center=False
    )
    img = utils.normalize_img(img)
    for edge in neurograph.mutable_edges:
        if neurograph.optimize_alignment or neurograph.optimize_path:
            xyz = to_img_coords(neurograph, edge)
            path = geometry_utils.sample_path(xyz, N_PROFILE_POINTS)
        else:
            i, j = tuple(edge)
            xyz_i = utils.world_to_img(neurograph, i)
            xyz_j = utils.world_to_img(neurograph, j)
            path = geometry_utils.make_line(xyz_i, xyz_j, N_PROFILE_POINTS)
        features[edge] = geometry_utils.get_profile(img, path, window=WINDOW)
    return features


def to_img_coords(neurograph, edge):
    """
    Converts xyz coordinate of each vertex in "edge" from real world to image
    coordinates.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    edge : frozenset
        The edge from "neurograph" to perform coordinate transformation.

    Returns
    -------
    img_coords : numpy.ndarray
        Image coordinates of each vertex from "edge".

    """
    xyz_list = neurograph.edges[edge]["xyz"]
    img_coords = [utils.world_to_img(neurograph, xyz) for xyz in xyz_list]
    return np.array(img_coords)


def generate_mutable_skel_features(neurograph):
    features = dict()
    for edge in neurograph.mutable_edges:
        i, j = tuple(edge)
        radius_i, radius_j = get_radii(neurograph, edge)
        dot1, dot2, dot3 = get_directionals(neurograph, edge, 5)
        ddot1, ddot2, ddot3 = get_directionals(neurograph, edge, 10)
        features[edge] = np.concatenate(
            (
                neurograph.compute_length(edge),
                neurograph.immutable_degree(i),
                neurograph.immutable_degree(j),
                radius_i,
                radius_j,
                dot1,
                dot2,
                dot3,
                ddot1,
                ddot2,
                ddot3,
            ),
            axis=None,
        )
    return features


def get_directionals(neurograph, edge, window):
    # Compute tangent vectors
    i, j = tuple(edge)
    edge_directional = geometry_utils.compute_tangent(
        neurograph.edges[edge]["xyz"]
    )
    directional_i = geometry_utils.get_directional(
        neurograph, i, edge_directional, window=window
    )
    directional_j = geometry_utils.get_directional(
        neurograph, j, edge_directional, window=window
    )

    # Compute features
    inner_product_1 = abs(np.dot(edge_directional, directional_i))
    inner_product_2 = abs(np.dot(edge_directional, directional_j))
    inner_product_3 = np.dot(directional_i, directional_j)
    return inner_product_1, inner_product_2, inner_product_3


def get_radii(neurograph, edge):
    i, j = tuple(edge)
    radius_i = neurograph.nodes[i]["radius"]
    radius_j = neurograph.nodes[j]["radius"]
    return radius_i, radius_j


# -- Build feature matrix
def get_feature_matrix(neurographs, features, model_type, block_ids=[]):
    assert model_type in SUPPORTED_MODELS, "Error! model_type not supported"
    if len(block_ids) > 0:
        return __multiblock_feature_matrix(
            neurographs, features, block_ids, model_type
        )
    else:
        return __feature_matrix(neurographs, features, model_type)


def __multiblock_feature_matrix(neurographs, features, blocks, model_type):
    # Initialize
    X = None
    y = None
    block_to_idxs = dict()
    idx_to_edge = dict()

    # Feature extraction
    for block_id in blocks:
        if len(neurographs[block_id].mutable_edges) == 0:
            block_to_idxs[block_id] = set()
            continue

        idx_shift = 0 if X is None else X.shape[0]
        if model_type == "MultiModalNet":
            X_i, x_i, y_i, idx_to_edge_i = get_multimodal_features(
                neurographs[block_id], features[block_id], shift=idx_shift
            )
        elif model_type == "ConvNet":
            X_i, y_i, idx_to_edge_i = get_img_chunks(
                neurographs[block_id], features[block_id], shift=idx_shift
            )
        else:
            X_i, y_i, idx_to_edge_i = get_feature_vectors(
                neurographs[block_id], features[block_id], shift=idx_shift
            )

        # Concatenate
        if X is None:
            X = deepcopy(X_i)
            y = deepcopy(y_i)
            if model_type == "MultiModalNet":
                x = deepcopy(x_i)
        else:
            X = np.concatenate((X, X_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)
            if model_type == "MultiModalNet":
                x = np.concatenate((x, x_i), axis=0)

        # Update dicts
        idxs = set(np.arange(idx_shift, idx_shift + len(idx_to_edge_i)))
        block_to_idxs[block_id] = idxs
        idx_to_edge.update(idx_to_edge_i)

    if model_type == "MultiModalNet":
        X = {"imgs": X, "features": x}

    return X, y, block_to_idxs, idx_to_edge


def __feature_matrix(neurographs, features, model_type):
    if model_type == "MultiModalNet":
        return get_multimodal_features(neurographs, features)
    elif model_type == "ConvNet":
        return get_img_chunks(neurographs, features)
    else:
        return get_feature_vectors(neurographs, features)


def get_feature_vectors(neurograph, features, shift=0):
    # Extract info
    features = combine_features(features)
    features.keys()
    key = sample(list(features.keys()), 1)[0]
    n_edges = neurograph.num_mutables()
    n_features = len(features[key])

    # Build
    idx_to_edge = dict()
    X = np.zeros((n_edges, n_features))
    y = np.zeros((n_edges))
    for i, edge in enumerate(features.keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features[edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, y, idx_to_edge


def get_multimodal_features(neurograph, features, shift=0):
    idx_to_edge = dict()
    n_edges = neurograph.num_mutables()
    X = np.zeros(((n_edges, 2) + tuple(CHUNK_SIZE)))
    x = np.zeros((n_edges, N_SKEL_FEATURES + N_PROFILE_POINTS))
    y = np.zeros((n_edges))
    for i, edge in enumerate(features["img_chunks"].keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features["img_chunks"][edge]
        x[i, :] = np.concatenate(
            (features["skel"][edge], features["img_profile"][edge])
        )
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, x, y, idx_to_edge


def get_img_chunks(neurograph, features, shift=0):
    idx_to_edge = dict()
    n_edges = neurograph.num_mutables()
    X = np.zeros(((n_edges, 2) + tuple(CHUNK_SIZE)))
    y = np.zeros((n_edges))
    for i, edge in enumerate(features["img_chunks"].keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features["img_chunks"][edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, y, idx_to_edge


# -- Utils --
def count_features(model_type):
    """
    Counts number of features based on the "model_type".

    Parameters
    ----------
    model_type : str
        Indication of model to be trained. Options include: AdaBoost,
        RandomForest, FeedForwardNet, ConvNet, MultiModalNet.

    Returns
    -------
    int
        Number of features.

    """
    if model_type != "ConvNet":
        return N_SKEL_FEATURES + N_PROFILE_POINTS


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
