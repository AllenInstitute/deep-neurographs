"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from google.cloud import storage

from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.swc_utils import process_gsc_zip, process_local_paths

# Graph construction
MIN_SIZE = 35
SMOOTH = True
PRUNE_CONNECTORS = False
PRUNE_SPURIOUS = True
PRUNE_DEPTH = 10
CONNECTOR_LENGTH = 16

# Proposal generation
N_PROPOSALS_PER_LEAF = 3
OPTIMIZE_PROPOSALS = False
OPTIMIZATION_DEPTH = 15
SEARCH_RADIUS = 10


# --- Build graph wrappers ---
def build_neurograph_from_local(
    anisotropy=[1.0, 1.0, 1.0],
    swc_dir=None,
    swc_paths=None,
    img_patch_origin=None,
    img_patch_shape=None,
    img_path=None,
    min_size=MIN_SIZE,
    progress_bar=False,
    prune_connectors=PRUNE_CONNECTORS,
    prune_spurious=PRUNE_SPURIOUS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    smooth=SMOOTH,
):
    # Process swc files
    assert swc_dir or swc_paths, "Provide swc_dir or swc_paths!"
    img_bbox = utils.get_img_bbox(img_patch_origin, img_patch_shape)
    paths = get_paths(swc_dir) if swc_dir else swc_paths
    swc_dicts, paths = process_local_paths(
        paths, anisotropy=anisotropy, min_size=min_size, img_bbox=img_bbox
    )

    # Build neurograph
    neurograph = build_neurograph(
        swc_dicts,
        img_bbox=img_bbox,
        img_path=img_path,
        swc_paths=paths,
        progress_bar=progress_bar,
        prune_connectors=prune_connectors,
        prune_spurious=prune_spurious,
        connector_length=connector_length,
        prune_depth=prune_depth,
        smooth=smooth,
    )

    # Delete nodes outside bbox
    if img_bbox:
        neurograph.delete_isolated()

    return neurograph


def build_neurograph_from_gcs_zips(
    bucket_name,
    cloud_path,
    anisotropy=[1.0, 1.0, 1.0],
    img_path=None,
    min_size=MIN_SIZE,
    n_proposals_per_leaf=N_PROPOSALS_PER_LEAF,
    prune_connectors=PRUNE_CONNECTORS,
    prune_spurious=PRUNE_SPURIOUS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    optimize_proposals=OPTIMIZE_PROPOSALS,
    optimization_depth=OPTIMIZATION_DEPTH,
    smooth=SMOOTH,
):
    """
    Builds a neurograph from a GCS bucket that contain of zips of swc files.

    Parameters
    ----------
    bucket_name : str
        Name of GCS bucket where zips are stored.
    cloud_path : str
        Path within GCS bucket to directory containing zips.
    img_path : str, optional
        Path to image stored GCS Bucket that swc files were generated from.
        The default is None.
    min_size : int, optional
        Minimum path length of swc files which are stored. The default is the
        global variable "MIN_SIZE".
    n_proposals_per_leaf : int, optional
        Number of edge proposals generated from each leaf node in an swc file.
        The default is the global variable "N_PROPOSALS_PER_LEAF".
    prune_connectors : bool, optional
        ...
    prune_spurious : bool, optional
        Indication of whether to prune short branches (i.e spurious branches).
        The default is the global variable "PRUNE".
    connector_length : int, optional
        ...
    prune_depth : int, optional
        Branches less than "prune_depth" microns are pruned if "prune" is
        True. The default is the global variable "PRUNE_DEPTH".
    optimize_proposals : bool, optional
        Indication of whether to optimize alignment of edge proposals to image
        signal. The default is the global variable "OPTIMIZE_PROPOSALS".
    optimization_depth : int, optional
        Distance from each edge proposal end point that is search during
        alignment optimization. The default is the global variable
        "OPTIMIZATION_DEPTH".
    smooth : bool, optional
        Indication of whether to smooth branches from swc files. The default
        is the global variable "SMOOTH".

    Returns
    -------
    neurograph : NeuroGraph
        Neurograph generated from zips of swc files stored in a GCS bucket.

    """
    # Process swc files
    print("Process swc files...")
    total_runtime, t0 = utils.init_timers()
    swc_dicts = download_gcs_zips(
        bucket_name, cloud_path, min_size, anisotropy
    )
    t, unit = utils.time_writer(time() - t0)
    print(f"\nModule Runtime: {round(t, 4)} {unit} \n")

    # Build neurograph
    print("Build NeuroGraph...")
    t0 = time()
    neurograph = build_neurograph(
        swc_dicts,
        img_path=img_path,
        prune_connectors=prune_connectors,
        prune_spurious=prune_spurious,
        connector_length=connector_length,
        prune_depth=prune_depth,
        smooth=smooth,
    )
    t, unit = utils.time_writer(time() - t0)
    print(f"Module Runtime: {round(t, 4)} {unit} \n")

    t, unit = utils.time_writer(time() - total_runtime)
    print(f"Total Runtime: {round(t, 4)} {unit}")
    print(f"Memory Consumption: {round(utils.get_memory_usage(), 4)} GBs")
    return neurograph


# -- Read swc files --
def download_gcs_zips(bucket_name, cloud_path, min_size, anisotropy):
    """
    Downloads swc files from zips stored in a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of GCS bucket where zips are stored.
    cloud_path : str
        Path within GCS bucket to directory containing zips.
    min_size : int
        Minimum path length of swc files which are stored.

    Returns
    -------
    swc_dicts : list

    """
    # Initializations
    bucket = storage.Client().bucket(bucket_name)
    zip_paths = utils.list_gcs_filenames(bucket, cloud_path, ".zip")
    chunk_size = int(len(zip_paths) * 0.02)

    # Parse
    cnt = 1
    t0, t1 = utils.init_timers()
    swc_dicts = []
    for i, path in enumerate(zip_paths):
        swc_dicts.extend(
            process_gsc_zip(
                bucket, path, anisotropy=anisotropy, min_size=min_size
            )
        )
        if i > cnt * chunk_size:
            cnt, t1 = report_progress(
                i, len(zip_paths), chunk_size, cnt, t0, t1
            )
    return swc_dicts


def count_files_in_zips(bucket, zip_paths):
    file_cnt = 0
    for zip_path in zip_paths:
        zip_content = bucket.blob(zip_path).download_as_bytes()
        file_cnt += len(utils.list_files_in_gcs_zip(zip_content))
    return file_cnt


# -- Build neurograph ---
def build_neurograph(
    swc_dicts,
    img_bbox=None,
    img_path=None,
    swc_paths=None,
    progress_bar=True,
    prune_connectors=PRUNE_CONNECTORS,
    prune_spurious=PRUNE_SPURIOUS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    smooth=SMOOTH,
):
    # Extract irreducibles
    n_components = len(swc_dicts)
    if progress_bar:
        print("(1) Extract irreducible nodes and edges")
        print("# connected components:", utils.reformat_number(n_components))
    irreducibles, n_nodes, n_edges = get_irreducibles(
        swc_dicts,
        progress_bar=progress_bar,
        prune_connectors=prune_connectors,
        prune_spurious=prune_spurious,
        connector_length=connector_length,
        prune_depth=prune_depth,
        smooth=smooth,
    )

    # Build neurograph
    if progress_bar:
        print("\n(2) Combine irreducibles...")
        print("# nodes:", utils.reformat_number(n_nodes))
        print("# edges:", utils.reformat_number(n_edges))

    neurograph = NeuroGraph(
        img_bbox=img_bbox, img_path=img_path, swc_paths=swc_paths
    )
    t0, t1 = utils.init_timers()
    chunk_size = max(int(n_components * 0.05), 1)
    cnt, i = 1, 0
    while len(irreducibles):
        irreducible_set = irreducibles.pop()
        neurograph.add_component(irreducible_set)
        if i > cnt * chunk_size and progress_bar:
            cnt, t1 = report_progress(i, n_components, chunk_size, cnt, t0, t1)
        i += 1
    if progress_bar:
        t, unit = utils.time_writer(time() - t0)
        print("\n" + f"add_irreducibles(): {round(t, 4)} {unit}")
    return neurograph


def get_irreducibles(
    swc_dicts,
    progress_bar=True,
    prune_connectors=PRUNE_CONNECTORS,
    prune_spurious=PRUNE_SPURIOUS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    smooth=SMOOTH,
):
    n_components = len(swc_dicts)
    chunk_size = max(int(n_components * 0.02), 1)
    with ProcessPoolExecutor() as executor:
        # Assign Processes
        processes = [None] * n_components
        i = 0
        while swc_dicts:
            swc_dict = swc_dicts.pop()
            processes[i] = executor.submit(
                gutils.get_irreducibles,
                swc_dict,
                prune_connectors,
                prune_spurious,
                prune_depth,
                smooth,
            )
            i += 1

        # Store results
        t0, t1 = utils.init_timers()
        n_nodes, n_edges = 0, 0
        progress_cnt = 1
        irreducibles = []
        connector_centroids = []
        for i, process in enumerate(as_completed(processes)):
            irreducibles_i, connector_centroids_i = process.result()
            irreducibles.extend(irreducibles_i)
            connector_centroids.extend(connector_centroids_i)
            n_nodes += count_nodes(irreducibles_i)
            n_edges += count_edges(irreducibles_i)
            if i > progress_cnt * chunk_size and progress_bar:
                progress_cnt, t1 = report_progress(
                    i, n_components, chunk_size, progress_cnt, t0, t1
                )
    if progress_bar:
        t, unit = utils.time_writer(time() - t0)
        print("\n" + f"get_irreducibles(): {round(t, 4)} {unit}")
    return irreducibles, n_nodes, n_edges


def count_nodes(irreducibles):
    n_nodes = 0
    for irreducibles_i in irreducibles:
        n_nodes += len(irreducibles_i["leafs"])
        n_nodes += len(irreducibles_i["junctions"])
    return n_nodes


def count_edges(irreducibles):
    n_edges = 0
    for irreducibles_i in irreducibles:
        n_edges += len(irreducibles_i["edges"])
    return n_edges


# -- Utils --
def get_paths(swc_dir):
    paths = []
    for f in utils.listdir(swc_dir, ext=".swc"):
        paths.append(os.path.join(swc_dir, f))
    return paths


def report_progress(current, total, chunk_size, cnt, t0, t1):
    eta = get_eta(current, total, chunk_size, t1)
    runtime = get_runtime(current, total, chunk_size, t0, t1)
    utils.progress_bar(current, total, eta=eta, runtime=runtime)
    return cnt + 1, time()


def get_eta(current, total, chunk_size, t0, return_str=True):
    chunk_runtime = time() - t0
    remaining = total - current
    eta = remaining * (chunk_runtime / chunk_size)
    t, unit = utils.time_writer(eta)
    return f"{round(t, 4)} {unit}" if return_str else eta


def get_runtime(current, total, chunk_size, t0, t1):
    eta = get_eta(current, total, chunk_size, t1, return_str=False)
    total_runtime = time() - t0 + eta
    t, unit = utils.time_writer(total_runtime)
    return f"{round(t, 4)} {unit}"
