"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features.

"""

from copy import deepcopy
from random import sample

import numpy as np

from deep_neurographs import geometry_utils, utils

CHUNK_SIZE = [64, 64, 64]
BUFFER = 256
HALF_CHUNK_SIZE = [CHUNK_SIZE[i] // 2 for i in range(3)]
WINDOW_SIZE = [5, 5, 5]

NUM_POINTS = 10
NUM_IMG_FEATURES = NUM_POINTS
NUM_SKEL_FEATURES = 11


# -- Wrappers --
def generate_mutable_features(
    neurograph,
    anisotropy=[1.0, 1.0, 1.0],
    img_profile=True,
    img_path=None,
    labels_path=None,
):
    """
    Generates feature vectors for every mutable edge in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    anisotropy : list[float]
        Real-world to image coordinates scaling factor for (x, y, z).
    img_path : str, optional
        Path to image volume.

    Returns
    -------
    Dictionary where each key-value pair corresponds to a type of feature
    vector and the numerical vector.

    """
    features = {"skel": generate_mutable_skel_features(neurograph)}
    if img_path and labels_path:
        features["img"] = generate_mutable_img_chunk_features(
            neurograph, img_path, labels_path, anisotropy=anisotropy
        )
    elif img_path and img_profile:
        features["img"] = generate_mutable_img_profile_features(
            neurograph, img_path, anisotropy=anisotropy
        )
    return features


# -- Edge feature extraction --
def generate_mutable_img_chunk_features(
    neurograph, img_path, labels_path, anisotropy=[1.0, 1.0, 1.0]
):
    features = dict()
    shape = neurograph.shape
    origin = neurograph.bbox["min"]  # world coordinates
    origin = utils.apply_anisotropy(
        origin, anisotropy, return_int=True
    )  # global image coordinates
    img, labels = utils.get_superchunks(
        img_path, labels_path, origin, shape, from_center=False
    )
    for edge in neurograph.mutable_edges:
        # Compute image coordinates
        edge_xyz = deepcopy(neurograph.edges[edge]["xyz"])
        edge_xyz = [
            utils.apply_anisotropy(
                edge_xyz[0] - origin, anisotropy=anisotropy
            ),
            utils.apply_anisotropy(
                edge_xyz[1] - origin, anisotropy=anisotropy
            ),
        ]

        # Extract chunks
        midpoint = geometry_utils.compute_midpoint(
            edge_xyz[0], edge_xyz[1]
        ).astype(int)
        img_chunk = utils.get_chunk(img, midpoint, CHUNK_SIZE)
        labels_chunk = utils.get_chunk(labels, midpoint, CHUNK_SIZE)

        # Compute path
        d = int(geometry_utils.dist(edge_xyz[0], edge_xyz[1]) + 5)
        img_coords_1 = np.round(
            edge_xyz[0] - midpoint + HALF_CHUNK_SIZE
        ).astype(int)
        img_coords_2 = np.round(
            edge_xyz[1] - midpoint + HALF_CHUNK_SIZE
        ).astype(int)
        path = geometry_utils.make_line(img_coords_1, img_coords_2, d)

        # Fill path
        labels_chunk[labels_chunk > 0] = 1
        labels_chunk = geometry_utils.fill_path(labels_chunk, path, val=-1)
        features[edge] = np.stack([img_chunk, labels_chunk], axis=0)

    return features


def get_chunk(superchunk, xyz):
    return deepcopy(
        superchunk[
            (xyz[0] - CHUNK_SIZE[0] // 2) : xyz[0] + CHUNK_SIZE[0] // 2,
            (xyz[1] - CHUNK_SIZE[1] // 2) : xyz[1] + CHUNK_SIZE[1] // 2,
            (xyz[2] - CHUNK_SIZE[2] // 2) : xyz[2] + CHUNK_SIZE[2] // 2,
        ]
    )


def generate_mutable_img_profile_features(
    neurograph, path, anisotropy=[1.0, 1.0, 1.0]
):
    features = dict()
    origin = utils.apply_anisotropy(
        neurograph.bbox["min"], anisotropy, return_int=True
    )
    shape = [neurograph.shape[i] + BUFFER for i in range(3)]
    superchunk = utils.get_superchunk(
        path, "zarr", origin, shape, from_center=False
    )
    for edge in neurograph.mutable_edges:
        edge_xyz = deepcopy(neurograph.edges[edge]["xyz"])
        edge_xyz = [
            utils.apply_anisotropy(
                edge_xyz[0] - neurograph.origin, anisotropy=anisotropy
            ),
            utils.apply_anisotropy(
                edge_xyz[1] - neurograph.origin, anisotropy=anisotropy
            ),
        ]
        line = geometry_utils.make_line(edge_xyz[0], edge_xyz[1], NUM_POINTS)
        features[edge] = geometry_utils.get_profile(
            superchunk, line, window_size=WINDOW_SIZE
        )
    return features


def generate_mutable_skel_features(neurograph):
    features = dict()
    for edge in neurograph.mutable_edges:
        i, j = tuple(edge)
        deg_i = len(list(neurograph.neighbors(i)))
        deg_j = len(list(neurograph.neighbors(j)))
        length = compute_length(neurograph, edge)
        radius_i, radius_j = get_radii(neurograph, edge)
        dot1, dot2, dot3 = get_directionals(neurograph, edge, 5)
        ddot1, ddot2, ddot3 = get_directionals(neurograph, edge, 10)
        features[edge] = np.concatenate(
            (
                length,
                deg_i,
                deg_j,
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


def compute_length(neurograph, edge, metric="l2"):
    i, j = tuple(edge)
    xyz_1, xyz_2 = neurograph.get_edge_attr("xyz", i, j)
    return geometry_utils.dist(xyz_1, xyz_2, metric=metric)


def get_directionals(neurograph, edge, window_size):
    # Compute tangent vectors
    i, j = tuple(edge)
    mutable_xyz_i, mutable_xyz_j = neurograph.get_edge_attr("xyz", i, j)
    mutable_xyz = np.array([mutable_xyz_i, mutable_xyz_j])
    mutable_tangent = geometry_utils.compute_tangent(mutable_xyz)
    context_tangent_i = geometry_utils.compute_context_vec(
        neurograph, i, mutable_tangent, window_size=window_size
    )
    context_tangent_j = geometry_utils.compute_context_vec(
        neurograph, j, mutable_tangent, window_size=window_size
    )

    # Compute features
    inner_product_1 = abs(np.dot(mutable_tangent, context_tangent_i))
    inner_product_2 = abs(np.dot(mutable_tangent, context_tangent_j))
    inner_product_3 = np.dot(context_tangent_i, context_tangent_j)
    return inner_product_1, inner_product_2, inner_product_3


def get_radii(neurograph, edge):
    i, j = tuple(edge)
    radius_i = neurograph.nodes[i]["radius"]
    radius_j = neurograph.nodes[j]["radius"]
    return radius_i, radius_j


# -- Build feature matrix
def build_feature_matrix(
    neurographs, features, blocks, img_chunks=False, multimodal=False
):
    # Initialize
    X = None
    y = None
    block_to_idxs = dict()
    idx_to_edge = dict()

    # Feature extraction
    for block_id in blocks:
        # Get features
        idx_shift = 0 if X is None else X.shape[0]
        if multimodal:
            X_i, x_i, y_i, idx_to_edge_i = build_multimodal_submatrix(
                neurographs[block_id], features[block_id], idx_shift
            )
        elif img_chunks:
            X_i, y_i, idx_to_edge_i = build_img_chunk_submatrix(
                neurographs[block_id], features[block_id], idx_shift
            )
        else:
            X_i, y_i, idx_to_edge_i = build_feature_submatrix(
                neurographs[block_id], features[block_id], idx_shift
            )

        # Concatenate
        if X is None:
            X = deepcopy(X_i)
            y = deepcopy(y_i)
            if multimodal:
                x = deepcopy(x_i)
        else:
            X = np.concatenate((X, X_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)
            if multimodal:
                x = np.concatenate((x, x_i), axis=0)

        # Update dicts
        idxs = set(np.arange(idx_shift, idx_shift + len(idx_to_edge_i)))
        block_to_idxs[block_id] = idxs
        idx_to_edge.update(idx_to_edge_i)

    if multimodal:
        X = {"imgs": X, "features": x}

    return X, y, block_to_idxs, idx_to_edge


def build_feature_submatrix(neurograph, features, shift):
    # Extract info
    features = combine_features(features)
    key = sample(list(features.keys()), 1)[0]
    num_edges = neurograph.num_mutables()
    num_features = len(features[key])

    # Build
    idx_to_edge = dict()
    X = np.zeros((num_edges, num_features))
    y = np.zeros((num_edges))
    for i, edge in enumerate(features.keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features[edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, y, idx_to_edge


def build_multimodal_submatrix(neurograph, features, shift):
    idx_to_edge = dict()
    num_edges = neurograph.num_mutables()
    X = np.zeros(((num_edges, 2) + tuple(CHUNK_SIZE)))
    x = np.zeros((num_edges, NUM_SKEL_FEATURES))
    y = np.zeros((num_edges))
    for i, edge in enumerate(features["img"].keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features["img"][edge]
        x[i, :] = features["skel"][edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, x, y, idx_to_edge


def build_img_chunk_submatrix(neurograph, features, shift):
    idx_to_edge = dict()
    num_edges = neurograph.num_mutables()
    X = np.zeros(((num_edges, 2) + tuple(CHUNK_SIZE)))
    y = np.zeros((num_edges))
    for i, edge in enumerate(features["img"].keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features["img"][edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, y, idx_to_edge


# -- Utils --
def compute_num_features(skel_features=True, img_features=True):
    num_features = NUM_SKEL_FEATURES if skel_features else 0
    num_features += NUM_IMG_FEATURES if img_features else 0
    return num_features


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
