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

from deep_neurographs import geometry, utils

CHUNK_SIZE = [64, 64, 64]
WINDOW = [5, 5, 5]
N_BRANCH_PTS = 50
N_PROFILE_PTS = 10
N_SKEL_FEATURES = 23
SUPPORTED_MODELS = [
    "AdaBoost",
    "RandomForest",
    "FeedForwardNet",
    "ConvNet",
    "MultiModalNet",
    "GraphNeuralNet",
]


# -- Wrappers --
def run(
    neurograph,
    model_type,
    search_radius,
    img_path,
    labels_path=None,
    proposals=None,
):
    if "Graph" in model_type:
        features = dict()
        features["branches"] = run_on_branches(neurograph)
        features["proposals"] = run_on_proposals(
            neurograph,
            model_type,
            search_radius,
            img_path,
            labels_path=labels_path,
            proposals=proposals,
        )
    else:
        features = run_on_proposals(
            neurograph,
            model_type,
            search_radius,
            img_path,
            labels_path=labels_path,
            proposals=proposals,
        )
    return features


def run_on_proposals(
    neurograph,
    model_type,
    search_radius,
    img_path,
    labels_path=None,
    proposals=None,
):
    """
    Generates feature vectors for every proposal in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    model_type : str
        Type of model to be trained. Options include: AdaBoost, RandomForest,
        FeedForwardNet, ConvNet, MultiModalNet.
    search_radius : float
        Search radius used to generate proposals.
    img_path : str
        Path to raw image stored in a GCS bucket.
    labels_path : str, optional
        Path to predicted segmentation stored in a GCS bucket. The default is
        None.
    proposals : list[frozenset], optional
        List of proposals for which features will be generated. The
        default is None.

    Returns
    -------
    features : dict
        Dictionary where each key-value pair corresponds to a type of feature
        vector and the numerical vector.

    """
    # Initializations
    img_driver = "n5" if ".n5" in img_path else "zarr"
    img = utils.open_tensorstore(img_path, img_driver)
    if labels_path:
        labels_driver = "neuroglancer_precomputed"
        labels = utils.open_tensorstore(labels_path, labels_driver)

    # Generate features
    proposals = neurograph.get_proposals() if proposals is None else proposals
    features = {
        "skel": generate_skel_features(neurograph, proposals, search_radius)
    }
    if model_type in ["ConvNet", "MultiModalNet"]:
        assert labels_path, "Must provide label_path for model_type!"
        features["img_chunks"], features["img_profile"] = generate_img_chunks(
            neurograph, proposals, img, labels
        )
    else:
        features["img_profile"] = generate_img_profiles(
            neurograph, proposals, img
        )
    return features


def run_on_branches(neurograph):
    """
    Generates feature vectors for every edge in a neurograph.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.

    Returns
    -------
    features : dict
        Dictionary where each key-value pair corresponds to a type of feature
        vector and the numerical vector.

    """
    return {"skel": generate_branch_features(neurograph)}


# -- Proposal Feature Extraction --
def generate_img_chunks(neurograph, proposals, img, labels):
    """
    Generates an image chunk for each proposal such that the centroid of the
    image chunk is the midpoint of the proposal. Image chunks contain two
    channels: raw image and predicted segmentation.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    img : tensorstore.TensorStore
        Image stored in a GCS bucket.
    labels : tensorstore.TensorStore
        Predicted segmentation mask stored in a GCS bucket.
    proposals : list[frozenset], optional
        List of proposals for which features will be generated. The
        default is None.

    Returns
    -------
    features : dict
        Dictonary such that each pair is the proposal id and image chunk.

    """
    with ThreadPoolExecutor() as executor:
        # Assign Threads
        threads = [None] * len(proposals)
        for t, proposal in enumerate(proposals):
            xyz_0, xyz_1 = neurograph.proposal_xyz(proposal)
            coord_0 = utils.to_img(xyz_0)
            coord_1 = utils.to_img(xyz_1)
            threads[t] = executor.submit(
                get_img_chunk, img, labels, coord_0, coord_1, proposal
            )

        # Save result
        chunks = dict()
        profiles = dict()
        for thread in as_completed(threads):
            proposal, chunk, profile = thread.result()
            chunks[proposal] = chunk
            profiles[proposal] = profile
    return chunks, profiles


def get_img_chunk(img, labels, coord_0, coord_1, thread_id=None):
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
    path = geometry.make_line(patch_coord_0, patch_coord_1, N_PROFILE_PTS)
    profile = geometry.get_profile(img_chunk, path, window=WINDOW)
    labels_chunk[labels_chunk > 0] = 1
    labels_chunk = geometry.fill_path(labels_chunk, path, val=2)
    chunk = np.stack([img_chunk, labels_chunk], axis=0)

    # Output
    if thread_id:
        return thread_id, chunk, profile
    else:
        return chunk, profile


def generate_img_profiles(neurograph, proposals, img):
    """
    Generates an image intensity profile along each proposal by reading from
    an image on the cloud.

    Parameters
    ----------
    neurograph : NeuroGraph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    proposals : list[frozenset]
        List of proposals for which features will be generated.
    img : tensorstore.TensorStore
        Image stored in a GCS bucket.

    Returns
    -------
    features : dict
        Dictonary such that each pair is the proposal id and image intensity
        profile.

    """
    # Generate coordinates
    coords = dict()
    for i, proposal in enumerate(proposals):
        coords[proposal] = get_profile_coords(neurograph, proposal)

    # Generate profiles
    img_profiles = dict()
    with ThreadPoolExecutor() as executor:
        threads = []
        for e, coords_e in coords.items():
            threads.append(executor.submit(get_profile, img, e, coords_e))

    for thread in as_completed(threads):
        e, img_profile_e = thread.result()
        img_profiles[e] = img_profile_e
    return img_profiles


def get_profile_coords(neurograph, proposal):
    """
    Gets coordinates needed to compute an image intensity profile.

    Parameters
    ----------
    neurograph : NeuroGarph
        NeuroGraph generated from a directory of swcs generated from a
        predicted segmentation.
    proposal : frozenset
        Proposal that image intensity profile will be generated for.

    Returns
    -------
    coords : dict
        Coordinates needed to compute an image intensity profile.

    """
    # Compute coordinates
    xyz_0, xyz_1 = neurograph.proposal_xyz(proposal)
    coord_0 = utils.to_img(xyz_0)
    coord_1 = utils.to_img(xyz_1)

    # Store coordinates
    bbox = utils.get_minimal_bbox(coord_0, coord_1)
    coords = {
        "bbox": bbox,
        "start": [coord_0[i] - bbox["min"][i] for i in range(3)],
        "end": [coord_1[i] - bbox["min"][i] for i in range(3)],
    }
    return coords


def get_profile(img, proposal, coords):
    """
    Gets the image intensity profile for a given proposal.

    Parameters
    ----------
    img : tensorstore.TensorStore
        Image to be queried.
    proposal : frozenset
        Proposal that image profile corresponds to.

    Returns
    -------
    proposal : frozenset
        Proposal that image profile corresponds to.
    list[int]
        Image intensity profile.

    """
    chunk = utils.read_tensorstore_bbox(img, coords["bbox"])
    line = geometry.make_line(coords["start"], coords["end"], N_PROFILE_PTS)
    return proposal, [chunk[tuple(xyz)] / 100 for xyz in line]


def generate_skel_features(neurograph, proposals, search_radius):
    features = dict()
    for proposal in proposals:
        i, j = tuple(proposal)
        features[proposal] = np.concatenate(
            (
                1,  # edge type
                neurograph.proposal_length(proposal),
                neurograph.degree[i],
                neurograph.degree[j],
                n_nearby_leafs(neurograph, proposal, search_radius),
                get_radii(neurograph, proposal),
                get_avg_radii(neurograph, proposal),
                get_directionals(neurograph, proposal, 8),
                get_directionals(neurograph, proposal, 16),
                get_directionals(neurograph, proposal, 32),
                get_directionals(neurograph, proposal, 64),
            ),
            axis=None,
        )
    return features


def get_directionals(neurograph, proposal, window_size):
    # Compute tangent vectors
    i, j = tuple(proposal)
    proposal_direction = geometry.compute_tangent(
        neurograph.proposals[proposal]["xyz"]
    )
    origin = neurograph.proposal_midpoint(proposal)
    direction_i = geometry.get_directional(neurograph, i, origin, window_size)
    direction_j = geometry.get_directional(neurograph, j, origin, window_size)

    # Compute features
    inner_product_1 = abs(np.dot(proposal_direction, direction_i))
    inner_product_2 = abs(np.dot(proposal_direction, direction_j))
    inner_product_3 = np.dot(direction_i, direction_j)
    return np.array([inner_product_1, inner_product_2, inner_product_3])


def get_avg_radii(neurograph, proposal):
    i, j = tuple(proposal)
    radii_i = neurograph.get_branches(i, ignore_reducibles=True, key="radius")
    radii_j = neurograph.get_branches(j, ignore_reducibles=True, key="radius")
    return np.array([get_avg_radius(radii_i), get_avg_radius(radii_j)])


def get_avg_radius(radii_list):
    avg = 0
    for radii in radii_list:
        end = min(16, len(radii) - 1)
        avg += np.mean(radii[0:end]) / len(radii_list)
    return avg


def get_avg_branch_lens(neurograph, proposal):
    i, j = tuple(proposal)
    branches_i = neurograph.get_branches(i, ignore_reducibles=True)
    branches_j = neurograph.get_branches(j, ignore_reducibles=True)
    return np.array([get_branch_len(branches_i), get_branch_len(branches_j)])


def get_branch_len(branch_list):
    branch_len = 0
    for branch in branch_list:
        branch_len += len(branch) / len(branch_list)
    return branch_len


def get_radii(neurograph, proposal):
    i, j = tuple(proposal)
    radius_i = neurograph.nodes[i]["radius"]
    radius_j = neurograph.nodes[j]["radius"]
    return np.array([radius_i, radius_j])


def avg_branch_radii(neurograph, edge):
    return np.array([np.mean(neurograph.edges[edge]["radius"])])


def n_nearby_leafs(neurograph, proposal, r):
    xyz = neurograph.proposal_midpoint(proposal)
    leafs = neurograph.query_kdtree(xyz, 24, node_type="leaf")
    return len(leafs) - 1


def n_nearby_junctions(neurograph, proposal, r):
    i, j = tuple(proposal)
    n_hits_i = count_junctions(neurograph, i, r)
    n_hits_j = count_junctions(neurograph, j, r)
    return np.array([n_hits_i, n_hits_j])


def count_junctions(neurograph, i, r):
    n_hits = 0
    swc_id_i = neurograph.nodes[i]["xyz"]
    xyz_i = neurograph.nodes[i]["xyz"]
    for xyz in neurograph.query_kdtree(xyz_i, r, node_type="junction"):
        if swc_id_i == neurograph.xyz_to_swc(xyz):
            n_hits += 1
    return n_hits


# --- Edge Feature Generation --
def generate_branch_features(neurograph):
    features = dict()
    for edge in neurograph.edges:
        i, j = tuple(edge)
        features[frozenset(edge)] = np.concatenate(
            (
                -1,  # edge type
                -1,
                neurograph.degree[i],
                neurograph.degree[j],
                -1,
                get_radii(neurograph, edge),
                np.mean(neurograph.edges[i, j]["radius"]),
                np.mean(neurograph.edges[i, j]["radius"]),
                -1 * np.ones(12),
                -1 * np.ones((N_PROFILE_PTS + 2)),
            ),
            axis=None,
        )
    return features


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


# -- Build feature matrix
def get_matrix(neurographs, features, model_type, block_ids=None):
    assert model_type in SUPPORTED_MODELS, "Error! model_type not supported"
    if block_ids:
        return __multiblock_feature_matrix(
            neurographs, features, block_ids, model_type
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
        elif model_type == "ConvNet":
            X_i, y_i, idx_transforms_i = stack_img_chunks(
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
    elif model_type == "ConvNet":
        return stack_img_chunks(neurographs, features)
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
    for i, edge in enumerate(features["img_chunks"].keys()):
        X[i, :] = features["img_chunks"][edge]
        x[i, :] = np.concatenate(
            (features["skel"][edge], features["img_profile"][edge])
        )
        y[i] = 1 if edge in neurograph.target_edges else 0
        idx_transforms["block_to_idxs"].add(i + shift)
        idx_transforms["idx_to_edge"][i + shift] = edge
    return X, x, y, idx_transforms


def stack_img_chunks(neurograph, features, shift=0):
    # Initialize
    X = np.zeros(((neurograph.n_proposals(), 2) + tuple(CHUNK_SIZE)))
    y = np.zeros((neurograph.n_proposals()))
    idx_transforms = {"block_to_idxs": set(), "idx_to_edge": dict()}

    # Build
    for i, edge in enumerate(features["img_chunks"].keys()):
        X[i, :] = features["img_chunks"][edge]
        y[i] = 1 if edge in neurograph.target_edges else 0
        idx_transforms["block_to_idxs"].add(i + shift)
        idx_transforms["idx_to_edge"][i + shift] = edge
    return X, y, idx_transforms


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
        return N_SKEL_FEATURES + N_PROFILE_PTS + 2


def combine_features(features):
    combined = dict()
    for edge in features["skel"].keys():
        combined[edge] = None
        for key in features.keys():
            if combined[edge] is None:
                combined[edge] = deepcopy(features[key][edge])
            else:
                combined[edge] = np.append(
                    combined[edge], np.mean(features[key][edge])
                )
                combined[edge] = np.append(
                    combined[edge], np.std(features[key][edge])
                )
                combined[edge] = np.concatenate(
                    (combined[edge], features[key][edge])
                )
    return combined
