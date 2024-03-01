"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training a model and performing inference.

Conventions:   (1) "xyz" refers to a real world coordinate such as those from
                   an swc file.

               (2) "coord" refers to an image coordinate in a whole exaspim
                   image. Note that we try to avoid using "coord" to refer to
                   coordinate in a superchunk or image patch.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from random import sample

import numpy as np
import tensorstore as ts

from deep_neurographs import geometry
from deep_neurographs import utils

CHUNK_SIZE = [64, 64, 64]
WINDOW = [5, 5, 5]
N_PROFILE_POINTS = 10
N_SKEL_FEATURES = 18
SUPPORTED_MODELS = [
    "AdaBoost",
    "RandomForest",
    "FeedForwardNet",
    "ConvNet",
    "MultiModalNet",
]


# -- Wrappers --
def generate_features(
    neurograph, model_type, img_path=None, labels_path=None, proposals=None
):
    """
    Generates feature vectors for every edge proposal in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    model_type : str
        Type of model to be trained. Options include: AdaBoost, RandomForest,
        FeedForwardNet, ConvNet, MultiModalNet.
    img_path : str, optional
        Path to raw image. The default is None.
    labels_path : str, optional
        Path to predicted segmentation. The default is None.
    proposals : list[frozenset], optional
        List of edge proposals for which features will be generated. The
        default is None.

    Returns
    -------
    features : dict
        Dictionary where each key-value pair corresponds to a type of feature
        vector and the numerical vector.

    """
    # Initialize proposals
    if proposals is None:
        proposals = neurograph.get_proposals()

    # Generate features
    features = {"skel": generate_skel_features(neurograph, proposals)}
    if model_type in ["ConvNet", "MultiModalNet"]:
        features["img_chunks"], features["img_profile"] = generate_img_chunks(
            neurograph, proposals, img_path, labels_path
        )
    if model_type in ["AdaBoost", "RandomForest", "FeedForwardNet"]:
        features["img_profile"] = generate_img_profiles(
            neurograph, proposals, img_path
        )
    return features


# -- Edge feature extraction --
def generate_img_chunks(neurograph, proposals, img_path, labels_path):
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
    proposals : list[frozenset], optional
        List of edge proposals for which features will be generated. The
        default is None.

    Returns
    -------
    features : dict
        Dictonary such that each pair is the edge id and image chunk.

    """
    driver = "n5" if ".n5" in img_path else "zarr"
    img = utils.open_tensorstore(img_path, driver)
    labels = utils.open_tensorstore(labels_path, "neuroglancer_precomputed")
    with ThreadPoolExecutor() as executor:
        # Assign Threads
        threads = [None] * len(proposals)
        for t, edge in enumerate(proposals):
            xyz_0, xyz_1 = neurograph.proposal_xyz(edge)
            coord_0 = utils.to_img(xyz_0)
            coord_1 = utils.to_img(xyz_1)
            threads[t] = executor.submit(
                get_img_chunks, img, labels, coord_0, coord_1, edge
            )

        # Save result
        chunks = dict()
        profiles = dict()
        for thread in as_completed(threads):
            edge, chunk, profile = thread.result()
            chunks[edge] = chunk
            profiles[edge] = profile
    return chunks, profiles


def get_img_chunks(img, labels, coord_0, coord_1, thread_id=None):
    # Extract chunks
    midpoint = geometry.get_midpoint(coord_0, coord_1).astype(int)
    if type(img) == ts.TensorStore:
        img_chunk = utils.read_tensorstore(img, midpoint, CHUNK_SIZE)
        labels_chunk = utils.read_tensorstore(labels, midpoint, CHUNK_SIZE)
    else:
        img_chunk = utils.get_chunk(img, midpoint, CHUNK_SIZE)
        labels_chunk = utils.get_chunk(labels, midpoint, CHUNK_SIZE)

    # Coordinate transform
    img_chunk = utils.normalize_img(img_chunk)
    patch_coord_0 = utils.img_to_patch(coord_0, midpoint, CHUNK_SIZE)
    patch_coord_1 = utils.img_to_patch(coord_1, midpoint, CHUNK_SIZE)

    # Generate features
    path = geometry.make_line(patch_coord_0, patch_coord_1, N_PROFILE_POINTS)
    profile = geometry.get_profile(img_chunk, path, window=WINDOW)
    labels_chunk[labels_chunk > 0] = 1
    labels_chunk = geometry.fill_path(labels_chunk, path, val=2)
    chunk = np.stack([img_chunk, labels_chunk], axis=0)

    # Output
    if thread_id:
        return thread_id, chunk, profile
    else:
        return chunk, profile


def generate_img_profiles(neurograph, proposals, path):
    if False:  # neurograph.bbox:
        return generate_img_profiles_via_superchunk(
            neurograph, proposals, path
        )
    else:
        return generate_img_profiles_via_multithreads(
            neurograph, proposals, path
        )


def generate_img_profiles_via_multithreads(neurograph, proposals, path):
    profile_features = dict()
    driver = "n5" if ".n5" in path else "zarr"
    img = utils.open_tensorstore(path, driver)
    with ThreadPoolExecutor() as executor:
        # Assign threads
        threads = [None] * len(proposals)
        for i, edge in enumerate(proposals):
            xyz_0, xyz_1 = neurograph.proposal_xyz(edge)
            coord_0 = utils.to_img(xyz_0)
            coord_1 = utils.to_img(xyz_1)
            line = geometry.make_line(coord_0, coord_1, N_PROFILE_POINTS)
            threads[i] = executor.submit(geometry.get_profile, img, line, edge)

        # Store result
        for thread in as_completed(threads):
            edge, profile = thread.result()
            profile_features[edge] = profile
    return profile_features


def generate_img_profiles_via_superchunk(neurograph, proposals, path):
    """
    Generates an image intensity profile along each edge proposal by reading
    a single superchunk from cloud that contains all proposals.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    path : str
        Path to raw image.
    proposals : list[frozenset], optional
        List of edge proposals for which features will be generated. The
        default is None.

    Returns
    -------
    features : dict
        Dictonary such that each pair is the edge id and image intensity
        profile.

    """
    features = dict()
    driver = "n5" if ".n5" in path else "zarr"
    img = utils.get_superchunk(
        path, driver, neurograph.origin, neurograph.shape, from_center=False
    )
    img = utils.normalize_img(img)
    for edge in neurograph.proposals:
        xyz_0, xyz_1 = neurograph.proposal_xyz(edge)
        coord_0 = utils.to_img(xyz_0) - neurograph.origin
        coord_1 = utils.to_img(xyz_1) - neurograph.origin
        path = geometry.make_line(coord_0, coord_1, N_PROFILE_POINTS)
        features[edge] = geometry.get_profile(img, path, window=WINDOW)
    return features


def generate_skel_features(neurograph, proposals):
    features = dict()
    for edge in proposals:
        i, j = tuple(edge)
        features[edge] = np.concatenate(
            (
                neurograph.proposal_length(edge),
                neurograph.degree[i],
                neurograph.degree[j],
                get_radii(neurograph, edge),
                get_avg_radii(neurograph, edge),
                get_avg_branch_lens(neurograph, edge),
                get_directionals(neurograph, edge, 8),
                get_directionals(neurograph, edge, 16),
                get_directionals(neurograph, edge, 32),
            ),
            axis=None,
        )
    return features


def get_directionals(neurograph, edge, window):
    # Compute tangent vectors
    i, j = tuple(edge)
    edge_direction = geometry.compute_tangent(
        neurograph.proposals[edge]["xyz"]
    )
    direction_i = geometry.get_directional(
        neurograph, i, edge_direction, window=window
    )
    direction_j = geometry.get_directional(
        neurograph, j, edge_direction, window=window
    )

    # Compute features
    inner_product_1 = abs(np.dot(edge_direction, direction_i))
    inner_product_2 = abs(np.dot(edge_direction, direction_j))
    inner_product_3 = np.dot(direction_i, direction_j)
    return np.array([inner_product_1, inner_product_2, inner_product_3])


def get_avg_radii(neurograph, edge):
    i, j = tuple(edge)
    radii_i = neurograph.get_branches(i, key="radius")
    radii_j = neurograph.get_branches(j, key="radius")
    return np.array([get_avg_radius(radii_i), get_avg_radius(radii_j)])


def get_avg_radius(radii_list):
    avg = 0
    for radii in radii_list:
        end = min(16, len(radii) - 1)
        avg += np.mean(radii[0:end]) / len(radii_list)
    return avg


def get_avg_branch_lens(neurograph, edge):
    i, j = tuple(edge)
    branches_i = neurograph.get_branches(i, key="xyz")
    branches_j = neurograph.get_branches(j, key="xyz")
    return np.array([get_branch_len(branches_i), get_branch_len(branches_j)])


def get_branch_len(branch_list):
    branch_len = 0
    for branch in branch_list:
        branch_len += len(branch) / len(branch_list)
    return branch_len


def get_radii(neurograph, edge):
    i, j = tuple(edge)
    radius_i = neurograph.nodes[i]["radius"]
    radius_j = neurograph.nodes[j]["radius"]
    return np.array([radius_i, radius_j])


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
        if neurographs[block_id].n_proposals() == 0:
            block_to_idxs[block_id] = set()
            continue

        idx_shift = 0 if X is None else X.shape[0]
        if model_type == "MultiModalNet":
            X_i, x_i, y_i, idx_to_edge_i = get_multimodal_features(
                neurographs[block_id], features[block_id], shift=idx_shift
            )
        elif model_type == "ConvNet":
            X_i, y_i, idx_to_edge_i = stack_img_chunks(
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
                print("if")
                x = deepcopy(x_i)
        else:
            X = np.concatenate((X, X_i), axis=0)
            y = np.concatenate((y, y_i), axis=0)
            if model_type == "MultiModalNet":
                print("else")
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
        return stack_img_chunks(neurographs, features)
    else:
        return get_feature_vectors(neurographs, features)


def get_feature_vectors(neurograph, features, shift=0):
    # Extract info
    features = combine_features(features)
    features.keys()
    key = sample(list(features.keys()), 1)[0]

    # Build
    idx_to_edge = dict()
    X = np.zeros((neurograph.n_proposals(), len(features[key])))
    y = np.zeros((neurograph.n_proposals()))
    for i, edge in enumerate(features.keys()):
        idx_to_edge[i + shift] = edge
        X[i, :] = features[edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
    return X, y, idx_to_edge


def get_multimodal_features(neurograph, features, shift=0):
    idx_to_edge = dict()
    n_edges = neurograph.n_proposals()
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


def stack_img_chunks(neurograph, features, shift=0):
    idx_to_edge = dict()
    X = np.zeros(((neurograph.n_proposals(), 2) + tuple(CHUNK_SIZE)))
    y = np.zeros((neurograph.n_proposals()))
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
