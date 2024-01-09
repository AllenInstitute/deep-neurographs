"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import os
import concurrent.futures

from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from io import BytesIO
from deep_neurographs import swc_utils, utils
from deep_neurographs.neurograph import NeuroGraph
from time import time
from zipfile import ZipFile

N_PROPOSALS_PER_LEAF = 3
OPTIMIZE_ALIGNMENT = False
OPTIMIZE_DEPTH = 15
PRUNE = True
PRUNE_DEPTH = 16
SEARCH_RADIUS = 0
SIZE_THRESHOLD = 100
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
    size_threshold=SIZE_THRESHOLD,
    smooth=SMOOTH,
):
    assert utils.xor(swc_dir, swc_list), "Error: provide swc_dir or swc_paths"
    neurograph = NeuroGraph(
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
        size_threshold=size_threshold,
        smooth=smooth,
    )
    if search_radius > 0:
        neurograph.generate_proposals(
            n_proposals_per_leaf=n_proposals_per_leaf,
            search_radius=search_radius
        )
    return neurograph


def build_neurograph_from_gcs_zips(
    bucket_name,
    cloud_path,
    img_path=None,
    size_threshold=SIZE_THRESHOLD,
    n_proposals_per_leaf=N_PROPOSALS_PER_LEAF,
    search_radius=SEARCH_RADIUS,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    optimize_alignment=OPTIMIZE_ALIGNMENT,
    optimize_depth=OPTIMIZE_DEPTH,
    smooth=SMOOTH,
):
    neurograph = NeuroGraph(
        img_path=img_path,
        optimize_alignment=optimize_alignment,
        optimize_depth=optimize_depth,
    )
    neurograph = init_immutables_from_gcs_zips(
        neurograph,
        bucket_name,
        cloud_path,
        prune=prune,
        prune_depth=prune_depth,
        size_threshold=size_threshold,
        smooth=smooth,
    )
    if search_radius > 0:
        neurograph.generate_proposals(
            n_proposals_per_leaf=n_proposals_per_leaf,
            search_radius=search_radius
        )
    return neurograph


def init_immutables_from_local(
    neurograph,
    swc_dir=None,
    swc_paths=None,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    size_threshold=SIZE_THRESHOLD,
    smooth=SMOOTH,
):
    swc_paths = get_paths(swc_dir) if swc_dir else swc_paths
    for path in swc_paths:
        neurograph.ingest_swc_from_local(
            path,
            prune=True,
            prune_depth=16,
            smooth=smooth,
        )
    return neurograph


def get_paths(swc_dir):
    swc_paths = []
    for f in utils.listdir(swc_dir, ext=".swc"):
        paths.append(os.path.join(swc_dir, f))
    return paths


def init_immutables_from_gcs_zips(
    neurograph,
    bucket_name,
    cloud_path,
    prune=PRUNE,
    prune_depth=PRUNE_DEPTH,
    size_threshold=SIZE_THRESHOLD,
    smooth=SMOOTH,
):
    # Initializations
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    zip_paths = list_gcs_filenames(bucket, cloud_path, ".zip")
    n_swc_files = 2080791 #count_files_in_zips(bucket, zip_paths)
    chunk_size = int(n_swc_files * 0.05)
    print("# zip files:", len(zip_paths))
    print(f"# swc files: {utils.reformat_number(n_swc_files)} \n\n")

    # Parse
    cnt = 1
    t0 = time()
    t1 = time()
    n_files_completed = 0
    print(f"-- Starting Multithread Reads with chunk_size={chunk_size} -- \n")
    for path in zip_paths:
        # Add to neurograph
        swc_dicts = process_gcs_zip(
            bucket,
            path,
        )
        if smooth:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                swc_dicts = list(executor.map(swc_utils.smooth, swc_dicts))

        # Readout progress
        n_files_completed += len(swc_dicts)
        if n_files_completed > cnt * chunk_size:
            report_runtimes(
                n_swc_files,
                n_files_completed,
                chunk_size,
                time() - t1,
                time() - t0,
            )
            cnt += 1
            t1 = time()
    t, unit = utils.time_writer(time() - t0)
    print(f"Total Runtime: {round(t, 4)} {unit}")
    return neurograph


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
    with ZipFile(BytesIO(zip_content), 'r') as zip_file:
        return zip_file.namelist()


def list_gcs_filenames(bucket, cloud_path, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    """
    blobs = bucket.list_blobs(prefix=cloud_path)
    return [blob.name for blob in blobs if extension in blob.name]


def process_gcs_zip(bucket, zip_path):
    # Get filenames
    zip_blob = bucket.blob(zip_path)
    zip_content = zip_blob.download_as_bytes()
    swc_paths = list_files_in_gcs_zip(zip_content)

    # Read files
    t0 = time()
    swc_dicts = [None] * len(swc_paths)
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(swc_utils.parse_gcs_zip, zip_file, path)
                for path in swc_paths
            ]
            for i, result_i in enumerate(as_completed(results)):
                swc_dicts[i] = result_i.result()
    return swc_dicts


def report_runtimes(
    n_files,
    n_files_completed,
    chunk_size,
    chunk_runtime,
    total_runtime,
):
    n_files_remaining = n_files - n_files_completed
    file_rate = chunk_runtime / chunk_size
    eta = (total_runtime + n_files_remaining  * file_rate) / 60
    files_processed = f"{n_files_completed - chunk_size}-{n_files_completed}"
    print(f"Completed: {round(100 * n_files_completed / n_files, 2)}%")
    print(f"Runtime for Files : {files_processed} {round(chunk_runtime, 4)} seconds")
    print(f"File Processing Rate: {file_rate} seconds")
    print(f"Approximate Total Runtime: {round(eta, 4)} minutes")
    print("")
