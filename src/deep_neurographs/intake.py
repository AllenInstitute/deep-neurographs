"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from io import BytesIO
from time import time
from zipfile import ZipFile

from google.cloud import storage

from deep_neurographs import graph_utils as gutils
from deep_neurographs import swc_utils, utils
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.swc_utils import parse_gcs_zip, process_local_paths

N_PROPOSALS_PER_LEAF = 3
OPTIMIZE_PROPOSALS = False
OPTIMIZATION_DEPTH = 15
PRUNE = True
PRUNE_DEPTH = 16
SEARCH_RADIUS = 0
MIN_SIZE = 30
SMOOTH = True


# --- Build graph wrappers ---
def build_neurograph_from_local(
    swc_dir=None,
    swc_paths=None,
    img_patch_origin=None,
    img_patch_shape=None,
    img_path=None,
    min_size=MIN_SIZE,
    n_proposals_per_leaf=N_PROPOSALS_PER_LEAF,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    optimize_proposals=OPTIMIZE_PROPOSALS,
    optimization_depth=OPTIMIZATION_DEPTH,
    search_radius=SEARCH_RADIUS,
    smooth=SMOOTH,
):
    # Process swc files
    t0 = time()
    assert utils.xor(swc_dir, swc_paths), "Error: provide swc_dir or swc_paths"
    bbox = utils.get_bbox(img_patch_origin, img_patch_shape)
    paths = get_paths(swc_dir) if swc_dir else swc_paths
    swc_dicts = process_local_paths(paths, min_size, bbox=bbox)
    print(f"process_local_paths(): {time() - t0} seconds")

    # Build neurograph
    t0 = time()
    neurograph = build_neurograph(
        swc_dicts,
        bbox=bbox,
        img_path=img_path,
        prune=prune,
        prune_depth=prune_depth,
        smooth=smooth,
    )
    print(f"build_neurograph(): {time() - t0} seconds")

    # Generate proposals
    t0 = time()
    if search_radius > 0:
        neurograph.generate_proposals(
            search_radius,
            n_proposals_per_leaf=n_proposals_per_leaf,
            optimize=optimize_proposals,
            optimization_depth=optimization_depth,
        )
    print(f"generate_proposals(): {time() - t0} seconds")

    return neurograph


def build_neurograph_from_gcs_zips(
    bucket_name,
    cloud_path,
    img_path=None,
    min_size=MIN_SIZE,
    n_proposals_per_leaf=N_PROPOSALS_PER_LEAF,
    search_radius=SEARCH_RADIUS,
    prune=PRUNE,
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
    search_radius : float, optional
        Maximum Euclidean length of an edge proposal. The default is the
        global variable "SEARCH_RADIUS".
    prune : bool, optional
        Indication of whether to prune short branches. The default is the
        global variable "PRUNE".
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
    swc_dicts = download_gcs_zips(bucket_name, cloud_path, min_size)
    neurograph = build_neurograph(
        swc_dicts,
        img_path=img_path,
        prune=prune,
        prune_depth=prune_depth,
        smooth=smooth,
    )
    if search_radius > 0:
        neurograph.generate_proposals(
            search_radius, n_proposals_per_leaf=n_proposals_per_leaf
        )
    return neurograph


# -- Read swc files --
def download_gcs_zips(bucket_name, cloud_path, min_size):
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
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    zip_paths = list_gcs_filenames(bucket, cloud_path, ".zip")
    chunk_size = int(len(zip_paths) * 0.1)
    print(f"# zip files: {len(zip_paths)} \n\n")

    # Parse
    cnt = 1
    t0 = time()
    t1 = time()
    swc_dicts = []
    print(f"-- Starting Multithread Reads with chunk_size={chunk_size} -- \n")
    for i, path in enumerate(zip_paths):
        swc_dict_i = download_zip(bucket, path, min_size=min_size)
        swc_dicts.extend(swc_dict_i)
        if i > cnt * chunk_size:
            report_runtimes(len(zip_paths), i, chunk_size, t0, t1)
            t1 = time()
            cnt += 1
        break
    t, unit = utils.time_writer(time() - t0)
    print("# connected components:", len(swc_dicts))
    print(f"Download Runtime: {round(t, 4)} {unit}")
    return swc_dicts


def download_zip(bucket, zip_path, min_size=0):
    zip_blob = bucket.blob(zip_path)
    zip_content = zip_blob.download_as_bytes()
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(parse_gcs_zip, zip_file, path, min_size)
                for path in list_files_in_gcs_zip(zip_content)
            ]
            swc_dicts = [result.result() for result in as_completed(results)]
    return swc_dicts


def count_files_in_zips(bucket, zip_paths):
    file_cnt = 0
    for zip_path in zip_paths:
        zip_blob = bucket.blob(zip_path)
        zip_content = zip_blob.download_as_bytes()
        file_paths = list_files_in_gcs_zip(zip_content)
        file_cnt += len(file_paths)
    return file_cnt


def list_files_in_gcs_zip(zip_content):
    """
    Lists all files in a zip file stored in a GCS bucket.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_gcs_filenames(bucket, cloud_path, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    """
    blobs = bucket.list_blobs(prefix=cloud_path)
    return [blob.name for blob in blobs if extension in blob.name]


# -- Build neurograph ---
def build_neurograph_old(
    swc_dicts,
    bbox=None,
    img_path=None,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    smooth=SMOOTH,
):
    # Extract irreducibles
    t0 = time()
    irreducibles = dict()
    for key in swc_dicts.keys():
        irreducibles[key] = gutils.get_irreducibles(
            swc_dicts[key], prune=prune, depth=prune_depth, smooth=smooth
        )
    print(f"   --> get_irreducibles(): {time() - t0} seconds")

    # Build neurograph
    t0 = time()
    neurograph = NeuroGraph(bbox=bbox, img_path=img_path)
    for key in swc_dicts.keys():
        neurograph.add_immutables(swc_dicts[key], irreducibles[key])
    print(f"   --> add_irreducibles(): {time() - t0} seconds")
    return neurograph


def build_neurograph(
    swc_dicts,
    bbox=None,
    img_path=None,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    smooth=SMOOTH,
):
    # Extract irreducibles
    irreducibles = dict()
    with ProcessPoolExecutor() as executor:
        # Assign Processes
        processes = [None] * len(swc_dicts)
        for i, key in enumerate(swc_dicts.keys()):
            processes[i] = executor.submit(
                gutils.get_irreducibles,
                swc_dicts[key],
                key,
                prune,
                prune_depth,
                smooth,
            )
        for process in as_completed(processes):
            process_id, result = process.result()
            irreducibles[process_id] = result

    # Build neurograph
    t0 = time()
    neurograph = NeuroGraph(bbox=bbox, img_path=img_path)
    for key in swc_dicts.keys():
        neurograph.add_immutables(irreducibles[key], swc_dicts[key], key)
    print(f"   --> add_irreducibles(): {time() - t0} seconds")
    return neurograph


# -- Utils --
def get_paths(swc_dir):
    paths = []
    for f in utils.listdir(swc_dir, ext=".swc"):
        paths.append(os.path.join(swc_dir, f))
    return paths


def get_start_ids(swc_dicts):
    node_ids = []
    cnt = 0
    for swc_dict in swc_dicts:
        graph = swc_utils.to_graph(swc_dict)
        leafs, junctions = gutils.get_irreducibles(graph)
        node_ids.append(cnt)
        cnt += len(leafs) + len(junctions)
    return node_ids


def report_runtimes(
    n_files, n_files_completed, chunk_size, start, start_chunk
):
    runtime = time() - start
    chunk_runtime = time() - start_chunk
    n_files_remaining = n_files - n_files_completed
    rate = chunk_runtime / chunk_size
    eta = (runtime + n_files_remaining * rate) / 60
    files_processed = f"{n_files_completed - chunk_size}-{n_files_completed}"
    print(f"Completed: {round(100 * n_files_completed / n_files, 2)}%")
    print(
        f"Runtime for Zips {files_processed}: {round(chunk_runtime, 4)} seconds"
    )
    print(f"Zip Processing Rate: {rate} seconds")
    print(f"Approximate Total Runtime: {round(eta, 4)} minutes")
    print("")
