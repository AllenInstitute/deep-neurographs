"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

import numpy as np
from deep_neurographs import utils

NUM_IMG_FEATURES = 0
NUM_SKEL_FEATURES = 4
NUM_PC_FEATURES = 0


# -- Wrappers --
def generate_node_features(neurograph, img=True, pointcloud=True, skel=True):
    features = dict()
    if img:
        features["img"] = generate_img_features(neurograph)

    if skel:
        features["skel"] = generate_skel_features(neurograph)

    if pointcloud:
        features["pointcloud"] = generate_pointcloud_features(neurograph)
    return extract_feature_vec(features)


def generate_immutable_features(neurograph, img=True, pointcloud=True, skel=True):
    features = dict()
    if img:
        features["img"] = generate_img_features(neurograph)

    if skel:
        features["skel"] = generate_immutable_skel_features(neurograph)

    if pointcloud:
        features["pointcloud"] = generate_pointcloud_features(neurograph)
    return extract_feature_vec(features)


def generate_mutable_features(neurograph, img=True, pointcloud=True, skel=True):
    features = dict()
    if img:
        features["img"] = generate_img_features(neurograph)

    if skel:
        features["skel"] = generate_mutable_skel_features(neurograph)

    if pointcloud:
        features["pointcloud"] = generate_pointcloud_features(neurograph)
    return extract_feature_vec(features)


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
        output = _generate_node_skel_features(neurograph, node)
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
def generate_immutable_skel_features(neurograph):
    features = dict()
    for edge in neurograph.immutable_edges:
        features[edge] = _generate_immutable_skel_features(neurograph, edge)
    return features


def _generate_immutable_skel_features(neurograph, edge):
    mean_xyz = np.mean(neurograph.edges[edge]["xyz"], axis=0)
    mean_radius = np.mean(neurograph.edges[edge]["radius"], axis=0)
    path_length = len(neurograph.edges[edge]["radius"])
    return np.concatenate((mean_xyz, mean_radius, path_length), axis=None)


def generate_mutable_skel_features(neurograph):
    features = dict()
    for edge in neurograph.mutable_edges:
        features[edge] = _generate_mutable_skel_features(neurograph, edge)
    return features


def _generate_mutable_skel_features(neurograph, edge):
    mean_xyz = np.mean(neurograph.edges[edge]["xyz"], axis=0)
    edge_length = compute_length(neurograph, edge)
    return np.concatenate((mean_xyz, edge_length), axis=None)


# -- Utils --
def compute_num_features(features):
    num_features = 0
    for key in features.keys():
        num_features += features[key][0]
    return num_features


def extract_feature_vec(features,):
    feature_vec = None
    for key in features.keys():
        if feature_vec is None:
            feature_vec = features[key]
        else:
            feature_vec = np.concatenate((feature_vec, features[key]), axis=1)
    return feature_vec


def compute_length(neurograph, edge):
    xyz_1 = neurograph.edges[edge]["xyz"][0]
    xyz_2 = neurograph.edges[edge]["xyz"][1]
    return utils.dist(xyz_1, xyz_2)