"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates features for training a model and performing inference.

Conventions:   (1) "xyz" refers to a real world coordinate such as those from
                   an swc file.

               (2) "voxel" refers to an voxel coordinate in a whole exaspim
                   image.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from random import sample

import numpy as np
import tensorstore as ts

from deep_neurographs import geometry
from deep_neurographs.machine_learning.heterograph_feature_generation import (
    generate_hgnn_features,
)
from deep_neurographs.utils import img_util, util

CHUNK_SIZE = [64, 64, 64]
N_BRANCH_PTS = 50
N_PROFILE_PTS = 16
N_SKEL_FEATURES = 22


def run(
    neurograph,
    img,
    model_type,
    proposals_dict,
    radius,
    downsample_factor=1,
    labels=None,
):
    """
    Generates feature vectors that are used by a machine learning model to
    classify proposals.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    img : tensorstore.Tensorstore
        Image stored in a GCS bucket.
    model_type : str
        Type of machine learning model used to classify proposals.
    proposals_dict : dict
        Dictionary that contains the items (1) "proposals" which are the
        proposals from "neurograph" that features will be generated and
        (2) "graph" which is the computation graph used by the gnn.
    radius : float
        Search radius used to generate proposals.
    downsample_factor : int, optional
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into. The default is 0.
    labels : tensorstore.TensorStore, optional
        Segmentation mask stored in a GCS bucket. The default is None.

    Returns
    -------
    dict
        Feature vectors.

    """
    # Init leaf kd-tree (if applicable)
    if neurograph.leaf_kdtree is None:
        neurograph.init_kdtree(node_type="leaf")

    # Feature generation by type of machine learning model
    if "Hetero" in model_type:
        return generate_hgnn_features(
            neurograph, img, proposals_dict, radius, downsample_factor
        )
    elif "Graph" in model_type:
        return generate_gnn_features(
            neurograph, img, proposals_dict, radius, downsample_factor
        )
    else:
        return generate_features(
            neurograph, img, proposals_dict, radius, downsample_factor
        )


def generate_gnn_features(
    neurograph, img, proposals_dict, radius, downsample_factor
):
    """
    Generates feature vectors used by a graph neural net (gnn) to classify
    proposals.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    img : tensorstore.Tensorstore
        Image stored in a GCS bucket.
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
        Feature vectors.

    """
    features = {
        "edges": run_on_edges(neurograph, proposals_dict),
        "proposals": run_on_proposals(
            neurograph, img, proposals_dict["proposals"], radius
        ),
    }
    return features


def generate_features(
    neurograph, img, proposals_dict, radius, downsample_factor
):
    """
    Generates feature vectors that are used by a general machine learning model
    (e.g. random forest or feed forward neural network).

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    img : tensorstore.Tensorstore
        Image stored in a GCS bucket.
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
        Feature vectors.

    """
    features = {
        "proposals": run_on_proposals(
            neurograph,
            img,
            proposals_dict["proposals"],
            radius,
            downsample_factor,
        )
    }
    return features


# -- feature generation by graphical structure type --
def run_on_edges(neurograph, proposals_dict):
    """
    Generates feature vectors for every edge in computation graph.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that proposals in "proposals_dict" belong to.
    proposals_dict : dict
        Dictionary containing the computation graph used by gnn and proposals
        to be classified.

    Returns
    -------
    dict
        Dictionary whose keys are feature types (i.e. skeletal) and values are
        a dictionary that maps an edge id to the corresponding feature vector.

    """
    edge_features = dict()
    for edge in proposals_dict["graph"].edges:
        if frozenset(edge) not in proposals_dict["proposals"]:
            edge_features[frozenset(edge)] = np.concatenate(
                (1, np.zeros((33))), axis=None
            )
    return {"skel": edge_features}


def run_on_proposals(neurograph, img, proposals, radius, downsample_factor):
    """
    Generates feature vectors for a set of proposals in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    img : tensorstore.Tensorstore
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
        "profiles": proposal_profiles(
            neurograph, img, proposals, downsample_factor
        ),
    }
    return proposal_features


# -- part 1: proposal feature generation --
def proposal_profiles(neurograph, img, proposals, downsample_factor):
    """
    Generates an image intensity profile along each proposal by reading from
    an image on the cloud.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    img : tensorstore.TensorStore
        Image stored in a GCS bucket.
    proposals : list[frozenset]
        List of proposals for which features will be generated.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Dictonary such that each pair is the proposal id and image intensity
        profile.

    """
    with ThreadPoolExecutor() as executor:
        threads = []
        for proposal in proposals:
            xyz_1, xyz_2 = neurograph.proposal_xyz(proposal)
            specs = get_profile_specs(xyz_1, xyz_2, downsample_factor)
            threads.append(executor.submit(get_profile, img, specs, proposal))

        profiles = dict()
        for thread in as_completed(threads):
            profiles.update(thread.result())
    return profiles


def get_profile_specs(xyz_1, xyz_2, downsample_factor):
    """
    Gets image bounding box and voxel coordinates needed to compute an image
    profile.

    Parameters
    ----------
    xyz_1 : numpy.ndarray
        xyz coordinate of starting point of profile.
    xyz_2 : numpy.ndarray
        xyz coordinate of ending point of profile.
    downsample_factor : int
        Downsampling factor that accounts for which level in the image pyramid
        the voxel coordinates must index into.

    Returns
    -------
    dict
        Specifications needed to compute an image profile for a given
        proposal.

    """
    # Compute voxel coordinates
    voxel_1 = img_util.to_voxels(xyz_1, downsample_factor=downsample_factor)
    voxel_2 = img_util.to_voxels(xyz_2, downsample_factor=downsample_factor)

    # Store local coordinates
    bbox = img_util.get_minimal_bbox(np.vstack([voxel_1, voxel_2]), buffer=1)
    start = [voxel_1[i] - bbox["min"][i] for i in range(3)]
    end = [voxel_2[i] - bbox["min"][i] for i in range(3)]
    specs = {
        "bbox": bbox,
        "profile_path": geometry.make_line(start, end, N_PROFILE_PTS),
    }
    return specs


def get_profile(img, specs, profile_id):
    """
    Gets the image profile for a given proposal.

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image that profiles are generated from.
    specs : dict
        Dictionary that contains the image bounding box and coordinates of the
        image profile path.
    profile_id : frozenset
        ...

    Returns
    -------
    dict
        Dictionary that maps an id (e.g. node, edge, or proposal) to its image
        profile.

    """
    profile = img_util.read_profile(img, specs)
    avg, std = util.get_avg_std(profile)
    profile.extend([avg, std])
    return {profile_id: profile}


def proposal_skeletal(neurograph, proposals, radius):
    """
    Generates features from skeleton (i.e. graph) which are graph or
    geometry type features.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    proposals : list
        Proposals for which features will be generated
    radius : float
        Search radius used to generate proposals.

    Returns
    -------
    dict
        Features generated from skeleton.

    """
    features = dict()
    for proposal in proposals:
        i, j = tuple(proposal)
        features[proposal] = np.concatenate(
            (
                neurograph.proposal_length(proposal),
                neurograph.degree[i],
                neurograph.degree[j],
                len(neurograph.nodes[i]["proposals"]),
                len(neurograph.nodes[j]["proposals"]),
                neurograph.n_nearby_leafs(proposal, radius),
                neurograph.proposal_radii(proposal),
                neurograph.proposal_avg_radii(proposal),
                neurograph.proposal_directionals(proposal, 8),
                neurograph.proposal_directionals(proposal, 16),
                neurograph.proposal_directionals(proposal, 32),
                neurograph.proposal_directionals(proposal, 64),
            ),
            axis=None,
        )
    return features


# --- part 2: edge feature generation --
def compute_curvature(neurograph, edge):
    kappa = curvature(neurograph.edges[edge]["xyz"])
    n_pts = len(kappa)
    if n_pts <= N_BRANCH_PTS:
        sampled_kappa = np.zeros((N_BRANCH_PTS))
        sampled_kappa[0:n_pts] = kappa
    else:
        idxs = np.linspace(0, n_pts - 1, N_BRANCH_PTS).astype(int)
        sampled_kappa = kappa[idxs]
    return np.array(sampled_kappa)


def curvature(xyz_list):
    a = np.linalg.norm(xyz_list[1:-1] - xyz_list[:-2], axis=1)
    b = np.linalg.norm(xyz_list[2:] - xyz_list[1:-1], axis=1)
    c = np.linalg.norm(xyz_list[2:] - xyz_list[:-2], axis=1)
    s = 0.5 * (a + b + c)
    delta = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return 4 * delta / (a * b * c)


# -- build feature matrix --
def get_matrix(neurographs, features, model_type, sample_ids=None):
    if sample_ids:
        return __multiblock_feature_matrix(
            neurographs, features, sample_ids, model_type
        )
    else:
        return __feature_matrix(neurographs, features, model_type)


def __multiblock_feature_matrix(neurographs, features, blocks, model_type):
    # Initialize
    X = None
    y = None
    idx_transforms = {"block_to_idxs": dict(), "idx_to_edge": dict()}

    # Feature extraction
    for block_id in blocks:
        if neurographs[block_id].n_proposals() == 0:
            idx_transforms["block_to_idxs"][block_id] = set()
            continue

        idx_shift = 0 if X is None else X.shape[0]
        if model_type == "MultiModalNet":
            X_i, x_i, y_i, idx_transforms_i = get_multimodal_features(
                neurographs[block_id], features[block_id], shift=idx_shift
            )
        else:
            X_i, y_i, idx_transforms_i = get_feature_vectors(
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
        idx_transforms["block_to_idxs"][block_id] = idx_transforms_i[
            "block_to_idxs"
        ]
        idx_transforms["idx_to_edge"].update(idx_transforms_i["idx_to_edge"])

    if model_type == "MultiModalNet":
        X = {"imgs": X, "features": x}

    return X, y, idx_transforms


def __feature_matrix(neurographs, features, model_type):
    if model_type == "MultiModalNet":
        return get_multimodal_features(neurographs, features)
    else:
        return get_feature_vectors(neurographs, features)


def get_feature_vectors(neurograph, features, shift=0):
    # Initialize
    features = combine_features(features)
    key = sample(list(features.keys()), 1)[0]
    X = np.zeros((len(features.keys()), len(features[key])))
    y = np.zeros((len(features.keys())))
    idx_transforms = {"block_to_idxs": set(), "idx_to_edge": dict()}

    # Build
    for i, edge in enumerate(features.keys()):
        X[i, :] = features[edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
        idx_transforms["block_to_idxs"].add(i + shift)
        idx_transforms["idx_to_edge"][i + shift] = edge
    return X, y, idx_transforms


def get_multimodal_features(neurograph, features, shift=0):
    # Initialize
    n_edges = neurograph.n_proposals()
    X = np.zeros(((n_edges, 2) + tuple(CHUNK_SIZE)))
    x = np.zeros((n_edges, N_SKEL_FEATURES + N_PROFILE_PTS))
    y = np.zeros((n_edges))
    idx_transforms = {"block_to_idxs": set(), "idx_to_edge": dict()}

    # Build
    for i, edge in enumerate(features["chunks"].keys()):
        X[i, :] = features["chunks"][edge]
        x[i, :] = np.concatenate(
            (features["skel"][edge], features["profiles"][edge])
        )
        y[i] = 1 if edge in neurograph.target_edges else 0
        idx_transforms["block_to_idxs"].add(i + shift)
        idx_transforms["idx_to_edge"][i + shift] = edge
    return X, x, y, idx_transforms


def stack_chunks(neurograph, features, shift=0):
    # Initialize
    X = np.zeros(((neurograph.n_proposals(), 2) + tuple(CHUNK_SIZE)))
    y = np.zeros((neurograph.n_proposals()))
    idx_transforms = {"block_to_idxs": set(), "idx_to_edge": dict()}

    # Build
    for i, edge in enumerate(features["chunks"].keys()):
        X[i, :] = features["chunks"][edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
        idx_transforms["block_to_idxs"].add(i + shift)
        idx_transforms["idx_to_edge"][i + shift] = edge
    return X, y, idx_transforms


# -- util --
def count_features():
    """
    Counts number of features based on the "model_type".

    Parameters
    ----------
    None

    Returns
    -------
    int
        Number of features.
    """
    return N_SKEL_FEATURES + N_PROFILE_PTS + 2


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


def generate_chunks(neurograph, proposals, img, labels):
    """
    Generates an image chunk for each proposal such that the centroid of the
    image chunk is the midpoint of the proposal. Image chunks contain two
    channels: raw image and predicted segmentation.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "proposals" belong to.
    img : tensorstore.TensorStore
        Image stored in a GCS bucket.
    labels : tensorstore.TensorStore
        Predicted segmentation mask stored in a GCS bucket.
    proposals : list[frozenset], optional
        List of proposals for which features will be generated. The
        default is None.

    Returns
    -------
    dict
        Dictonary such that each pair is the proposal id and image chunk.

    """
    with ThreadPoolExecutor() as executor:
        # Assign Threads
        threads = [None] * len(proposals)
        for t, proposal in enumerate(proposals):
            xyz_0, xyz_1 = neurograph.proposal_xyz(proposal)
            voxel_1 = util.to_voxels(xyz_0)
            voxel_2 = util.to_voxels(xyz_1)
            threads[t] = executor.submit(
                get_chunk, img, labels, voxel_1, voxel_2, proposal
            )

        # Save result
        chunks = dict()
        profiles = dict()
        for thread in as_completed(threads):
            proposal, chunk, profile = thread.result()
            chunks[proposal] = chunk
            profiles[proposal] = profile
    return chunks, profiles


def get_chunk(img, labels, voxel_1, voxel_2, thread_id=None):
    # Extract chunks
    midpoint = geometry.get_midpoint(voxel_1, voxel_2).astype(int)
    if type(img) == ts.TensorStore:
        chunk = util.read_tensorstore(img, midpoint, CHUNK_SIZE)
        labels_chunk = util.read_tensorstore(labels, midpoint, CHUNK_SIZE)
    else:
        chunk = img_util.read_chunk(img, midpoint, CHUNK_SIZE)
        labels_chunk = img_util.read_chunk(labels, midpoint, CHUNK_SIZE)

    # Coordinate transform
    chunk = util.normalize(chunk)
    patch_voxel_1 = util.voxels_to_patch(voxel_1, midpoint, CHUNK_SIZE)
    patch_voxel_2 = util.voxels_to_patch(voxel_2, midpoint, CHUNK_SIZE)

    # Generate features
    path = geometry.make_line(patch_voxel_1, patch_voxel_2, N_PROFILE_PTS)
    profile = geometry.get_profile(chunk, path)
    labels_chunk[labels_chunk > 0] = 1
    labels_chunk = geometry.fill_path(labels_chunk, path, val=2)
    chunk = np.stack([chunk, labels_chunk], axis=0)

    # Output
    if thread_id:
        return thread_id, chunk, profile
    else:
        return chunk, profile
