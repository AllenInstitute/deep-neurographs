"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds graph for postprocessing with GNN.

"""

import numpy as np

NUM_EDGE_FEATURES = 1
NUM_IMG_FEATURES = 0
NUM_SKEL_FEATURES = 4
NUM_PC_FEATURES = 0


# -- Wrappers --
def generate_node_features(supergraph, img=True, pointcloud=True, skel=True):
    features = dict()
    if img:
        features["img"] = generate_img_features(supergraph)

    if skel:
        features["skel"] = generate_skel_features(supergraph)

    if pointcloud:
        features["pointcloud"] = generate_pointcloud_features(supergraph)

    return extract_feature_vec(features)


def generate_edge_features(supergraph):
    features = np.zeros((supergraph.num_edges(), NUM_EDGE_FEATURES))
    for i, edge in enumerate(supergraph.edges()):
        features[i] = supergraph.edges[edge]["distance"]
    return features


# -- Node feature extraction --
def generate_img_features(supergraph):
    img_features = np.zeros((supergraph.num_nodes(), NUM_IMG_FEATURES))
    for node in supergraph.nodes:
        img_features[node] = _generate_node_img_features()
    return img_features


def _generate_node_img_features():
    pass


def generate_skel_features(supergraph):
    skel_features = np.zeros((supergraph.num_nodes(), NUM_SKEL_FEATURES))
    for node in supergraph.nodes:
        skel_features[node] = _generate_node_skel_features(supergraph, node)
    return skel_features


def _generate_node_skel_features(supergraph, node):
    mean_radius = np.mean(supergraph.nodes[node]["radius"])
    mean_xyz = np.mean(supergraph.nodes[node]["xyz"], axis=0)
    return np.concatenate((mean_radius, mean_xyz), axis=None)


def generate_pointcloud_features(supergraph):
    pc_features = np.zeros((supergraph.num_nodes(), NUM_PC_FEATURES))
    for node in supergraph.nodes:
        pc_features[node] = _generate_pointcloud_node_features()
    return pc_features


def _generate_pointcloud_node_features():
    pass


# -- Edge feature extraction --


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
