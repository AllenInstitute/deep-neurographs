"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

from copy import deepcopy
from random import sample

import numpy as np
from scipy.linalg import svd

from deep_neurographs import geometry_utils, utils

NUM_IMG_FEATURES = 0
NUM_SKEL_FEATURES = 9
NUM_PC_FEATURES = 0


# -- Wrappers --
def generate_mutable_features(
    neurograph, img=True, pointcloud=True, skel=True
):
    features = dict()
    if img:
        features["img"] = generate_img_features(neurograph)
    if skel:
        features["skel"] = generate_mutable_skel_features(neurograph)
    features = combine_feature_vecs(features)
    return features


# -- Node feature extraction --
def generate_img_features(neurograph):
    img_features = np.zeros((neurograph.num_nodes(), NUM_IMG_FEATURES))
    for node in neurograph.nodes:
        img_features[node] = _generate_node_img_features()
    return img_features


def _generate_node_img_features():
    pass


def generate_skel_features(neurograph):
    skel_features = np.zeros((neurograph.num_nodes(), NUM_SKEL_FEATURES))
    for node in neurograph.nodes:
        skel_features[node] = _generate_node_skel_features(neurograph, node)
    return skel_features


def _generate_node_skel_features(neurograph, node):
    radius = neurograph.nodes[node]["radius"]
    xyz = neurograph.nodes[node]["xyz"]
    return np.append(xyz, radius)


def generate_pointcloud_features(neurograph):
    pc_features = np.zeros((neurograph.num_nodes(), NUM_PC_FEATURES))
    for node in neurograph.nodes:
        pc_features[node] = _generate_pointcloud_node_features()
    return pc_features


def _generate_pointcloud_node_features():
    pass


# -- Edge feature extraction --
def generate_mutable_skel_features(neurograph):
    features = dict()
    for edge in neurograph.mutable_edges:
        length = compute_length(neurograph, edge)
        radius_i, radius_j = get_radii(neurograph, edge)

        dot1, dot2, dot3 = get_directionals(neurograph, edge, 5)
        ddot1, ddot2, ddot3 = get_directionals(neurograph, edge, 5)
        features[edge] = np.concatenate((length, radius_i, radius_j, dot1, dot2, dot3), axis=None)
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


# -- Combine feature vectors
def build_feature_matrix(neurographs, features, blocks):
    # Initialize
    X = None
    block_to_idxs = dict()
    idx_to_edge = dict()

    # Feature extraction
    for block_id in blocks:
        # Get features
        idx_shift = 0 if X is None else X.shape[0]
        X_i, y_i, idx_to_edge_i = build_feature_submatrix(
            neurographs[block_id], features[block_id], idx_shift
        )

        # Concatenate
        if X is None:
            X = deepcopy(X_i)
            y = deepcopy(y_i)
        else:
            X = np.concatenate((X, X_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)

        # Update dicts
        idxs = set(np.arange(idx_shift, idx_shift + len(idx_to_edge_i)))
        block_to_idxs[block_id] = idxs
        idx_to_edge.update(idx_to_edge_i)
    return X, y, block_to_idxs, idx_to_edge


def build_feature_submatrix(neurograph, feat_dict, shift):
    # Extract info
    key = sample(list(feat_dict.keys()), 1)[0]
    num_edges = neurograph.num_mutables()
    num_features = len(feat_dict[key])

    # Build
    idx_to_edge = dict()
    X = np.zeros((num_edges, num_features))
    y = np.zeros((num_edges))
    for i, edge in enumerate(feat_dict.keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = feat_dict[edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, y, idx_to_edge


# -- Utils --
def compute_num_features(features):
    num_features = 0
    for key in features.keys():
        num_features += features[key][0]
    return num_features


def combine_feature_vecs(features):
    vec = None
    for key in features.keys():
        if vec is None:
            vec = features[key]
        else:
            vec = np.concatenate((vec, features[key]), axis=1)
    return vec


"""

def generate_node_features(neurograph, img=True, pointcloud=True, skel=True):
    features = dict()
    if img:
        features["img"] = generate_img_features(neurograph)

    if skel:
        features["skel"] = generate_skel_features(neurograph)

    if pointcloud:
        features["pointcloud"] = generate_pointcloud_features(neurograph)
    return extract_feature_vec(features)


def generate_immutable_features(
    neurograph, img=True, pointcloud=True, skel=True
):
    features = dict()
    if img:
        features["img"] = generate_img_features(neurograph)

    if skel:
        features["skel"] = generate_immutable_skel_features(neurograph)

    if pointcloud:
        features["pointcloud"] = generate_pointcloud_features(neurograph)
    return extract_feature_vec(features)

def generate_immutable_skel_features(neurograph):
    features = dict()
    for edge in neurograph.immutable_edges:
        features[edge] = _generate_immutable_skel_features(neurograph, edge)
    return features


def _generate_immutable_skel_features(neurograph, edge):
    mean_radius = np.mean(neurograph.edges[edge]["radius"], axis=0)
    return np.concatenate((mean_radius), axis=None)
"""
