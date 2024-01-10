"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import concurrent.futures
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from time import time
from zipfile import ZipFile

from google.cloud import storage

from deep_neurographs import graph_utils as gutils, swc_utils, utils
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.swc_utils import parse_gcs_zip

N_PROPOSALS_PER_LEAF = 3
OPTIMIZE_ALIGNMENT = False
OPTIMIZE_DEPTH = 15
PRUNE = True
PRUNE_DEPTH = 16
SEARCH_RADIUS = 0
MIN_SIZE = 30
SMOOTH = False


# --- Build graph ---
def build_neurograph_from_local(
    swc_dir=None,
    swc_paths=None,
    img_patch_shape=None,
    img_patch_origin=None,
    img_path=None,
    n_proposals_per_leaf=N_PROPOSALS_PER_LEAF,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    optimize_alignment=OPTIMIZE_ALIGNMENT,
    optimize_depth=OPTIMIZE_DEPTH,
    search_radius=SEARCH_RADIUS,
    min_size=MIN_SIZE,
    smooth=SMOOTH,
):
    assert utils.xor(swc_dir, swc_paths), "Error: provide swc_dir or swc_paths"
    neurograph = NeuroGraph(
        swc_dir=swc_dir,
        img_path=img_path,
        optimize_depth=optimize_depth,
        optimize_alignment=optimize_alignment,
        origin=img_patch_origin,
        shape=img_patch_shape,
    )
    neurograph = init_immutables_from_local(
        neurograph,
        swc_dir=swc_dir,
        swc_paths=swc_paths,
        prune=prune,
        prune_depth=prune_depth,
        min_size=min_size,
        smooth=smooth,
    )
    if search_radius > 0:
        neurograph.generate_proposals(
            n_proposals_per_leaf=n_proposals_per_leaf,
            search_radius=search_radius,
        )
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
    optimize_alignment=OPTIMIZE_ALIGNMENT,
    optimize_depth=OPTIMIZE_DEPTH,
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
    img_path : str
        Path to image stored GCS Bucket that swc files were generated from.
    min_size : int
        Minimum path length of swc files which are stored.
    n_proposals_per_leaf : int
        Number of edge proposals generated from each leaf node in an swc file.
    search_radius : float
        Maximum Euclidean length of an edge proposal.
    prune : bool
        Indication of whether to prune short branches.
    prune_depth : int
        Branches less than "prune_depth" microns are pruned if "prune" is
        True.
    optimize_alignment : bool
        Indication of whether to optimize alignment of edge proposals to image
        signal.
    optimize_depth : int
        Distance from each edge proposal end point that is search during
        alignment optimization. 
    smooth : bool
        Indication of whether to smooth branches from swc files.

    Returns
    -------
    neurograph : NeuroGraph
        Neurograph generated from zips of swc files stored in a GCS bucket.

    """
    swc_dicts = download_gcs_zips(bucket_name, cloud_path, min_size=min_size)
    neurograph = build_neurograph(
        swc_dicts,
        img_path=img_path,
        prune=prune,
        prune_depth=prune_depth,
        smooth=smooth,
        optimize_alignment=OPTIMIZE_ALIGNMENT,
        optimize_depth=OPTIMIZE_DEPTH,
    )
    if search_radius > 0:
        neurograph.generate_proposals(
            n_proposals_per_leaf=n_proposals_per_leaf,
            search_radius=search_radius,
        )
    return neurograph


def init_immutables_from_local(
    neurograph,
    swc_dir=None,
    swc_paths=None,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    min_size=MIN_SIZE,
    smooth=SMOOTH,
):
    swc_paths = get_paths(swc_dir) if swc_dir else swc_paths
    for path in swc_paths:
        neurograph.ingest_swc_from_local(
            path, prune=True, prune_depth=16, smooth=smooth
        )
    return neurograph


def get_paths(swc_dir):
    paths = []
    for f in utils.listdir(swc_dir, ext=".swc"):
        paths.append(os.path.join(swc_dir, f))
    return paths


def download_gcs_zips(
    bucket_name,
    cloud_path,
    min_size=0,
):
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
    print(f"# zip files: {len(zip_paths)} \n\n", )

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
    t0 = time()
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


def report_runtimes(
    n_files, n_files_completed, chunk_size, start, start_chunk,
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
    print(f"Zip Processing Rate: {file_rate} seconds")
    print(f"Approximate Total Runtime: {round(eta, 4)} minutes")
    print("")


def build_neurograph(
    swc_dicts,
    img_path=None,
    optimize_alignment=OPTIMIZE_ALIGNMENT,
    optimize_depth=OPTIMIZE_DEPTH,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    smooth=SMOOTH
):
    graph_list = build_graphs(swc_dicts, prune, prune_depth, smooth)
    start_ids = get_start_ids(swc_dicts)
    print("Total Runtime:", 1600 * t)
    stop
    neurograph = NeuroGraph(
        img_path=img_path,
        optimize_alignment=optimize_alignment,
        optimize_depth=optimize_depth,
        )

def build_graphs(swc_dicts, prune, prune_depth, smooth):
    t0 = time()
    graphs = [None] * len(swc_dicts)
    for i, swc_dict in enumerate(swc_dicts):
        graphs[i] = build_subgraph(swc_dict)
    t = time() - t0
    print(f"build_subgraphs(): {t} seconds")
    return graphs


def build_subgraph(swc_dict):
    graph = nx.Graph()
    graph.add_edges_from(zip(swc_dict["id"][1:], swc_dict["pid"][1:]))
    return graph
    

def get_start_ids(swc_dicts):
    # runtime: ~ 1 minute
    t0 = time()
    node_ids = []
    cnt = 0
    for swc_dict in swc_dicts:
        graph = swc_utils.to_graph(swc_dict)
        leafs, junctions = gutils.get_irreducibles(graph)
        node_ids.append(cnt)
        cnt += len(leafs) + len(junctions)
    return node_ids
