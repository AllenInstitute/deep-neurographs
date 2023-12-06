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
        features["img"] = generate_img_chunk_features(
            neurograph, img_path, labels_path, anisotropy=anisotropy
        )
    elif img_path and img_profile:
        features["img"] = generate_img_profile_features(
            neurograph, img_path, anisotropy=anisotropy
        )
    return features


# -- Edge feature extraction --
def generate_img_chunk_features(
    neurograph, img_path, labels_path, anisotropy=[1.0, 1.0, 1.0]
):
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
        midpoint = geometry_utils.compute_midpoint(xyz_i, xyz_j).astype(int)
        img_chunk = utils.get_chunk(img, midpoint, CHUNK_SIZE)
        labels_chunk = utils.get_chunk(labels, midpoint, CHUNK_SIZE)

        # Mark path
        if neurograph.optimize_proposals:
            xyz_list = to_patch_coords(neurograph, edge, midpoint)
            path = geometry_utils.sample_path(xyz_list, NUM_POINTS)
        else:
            d = int(geometry_utils.dist(xyz_i, xyz_j) + 5)
            img_coords_i = utils.img_to_patch(xyz_i, midpoint, HALF_CHUNK_SIZE)
            img_coords_j = utils.img_to_patch(xyz_j, midpoint, HALF_CHUNK_SIZE)
            path = geometry_utils.make_line(img_coords_i, img_coords_j, d)

        img_chunk = utils.normalize_img(img_chunk)
        labels_chunk[labels_chunk > 0] = 1
        labels_chunk = geometry_utils.fill_path(labels_chunk, path)
        features[edge] = np.stack([img_chunk, labels_chunk], axis=0)

    return features


def to_patch_coords(neurograph, edge, midpoint):
    patch_coord_list = []
    for xyz in neurograph.edges[edge]["xyz"]:
        img_coord = utils.world_to_img(neurograph, xyz)
        patch_coord = utils.img_to_patch(img_coord, midpoint, HALF_CHUNK_SIZE)
        patch_coord_list.append(patch_coord)
    return np.array(patch_coord_list[3:-3])


def generate_img_profile_features(
    neurograph, path, anisotropy=[1.0, 1.0, 1.0]
):
    features = dict()
    origin = utils.apply_anisotropy(neurograph.origin, return_int=True)
    img = utils.get_superchunk(
        path, "zarr", origin, neurograph.shape, from_center=False
    )
    img = utils.normalize_img(img)
    simple_edges = neurograph.get_simple_proposals()
    for edge in neurograph.mutable_edges:
        if neurograph.optimize_proposals and edge in simple_edges:
            xyz = to_img_coords(neurograph, edge)
            path = geometry_utils.sample_path(xyz, NUM_POINTS)
        else:
            i, j = tuple(edge)
            xyz_i = utils.world_to_img(neurograph, i)
            xyz_j = utils.world_to_img(neurograph, j)
            path = geometry_utils.make_line(xyz_i, xyz_j, NUM_POINTS)
        features[edge] = geometry_utils.get_profile(
            img, path, window_size=WINDOW_SIZE
        )
    return features


def to_img_coords(neurograph, edge):
    img_coords = []
    for xyz in neurograph.edges[edge]["xyz"]:
        img_coords.append(utils.world_to_img(neurograph, xyz))
    img_coords = np.array(img_coords)
    return img_coords[3:-3, :] if img_coords.shape[0] > 10 else img_coords


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
    tangent = geometry_utils.compute_tangent(neurograph.edges[edge]["xyz"])
    context_tangent_i = geometry_utils.get_directional(
        neurograph, i, tangent, window=window
    )
    context_tangent_j = geometry_utils.get_directional(
        neurograph, j, tangent, window=window
    )

    # Compute features
    inner_product_1 = abs(np.dot(tangent, context_tangent_i))
    inner_product_2 = abs(np.dot(tangent, context_tangent_j))
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
