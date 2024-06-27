"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from google.cloud import storage

from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.swc_utils import process_gcs_zip, process_local_paths

MIN_SIZE = 30
NODE_SPACING = 2
SMOOTH = True
PRUNE_CONNECTORS = False
PRUNE_DEPTH = 16
TRIM_DEPTH = 0
CONNECTOR_LENGTH = 8


# --- Build graph wrappers ---
def build_neurograph_from_local(
    anisotropy=[1.0, 1.0, 1.0],
    img_patch_origin=None,
    img_patch_shape=None,
    img_path=None,
    min_size=MIN_SIZE,
    node_spacing=NODE_SPACING,
    progress_bar=False,
    prune_connectors=PRUNE_CONNECTORS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    trim_depth=TRIM_DEPTH,
    smooth=SMOOTH,
    swc_dir=None,
    swc_paths=None,
):
    """
    Builds a neurograph from swc files on the local machine.

    Parameters
    ----------
    anisotropy : list[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope. The default is [1.0, 1.0, 1.0].
    image_patch_origin : list[float], optional
        An xyz coordinate in the image which is the upper, left, front corner
        of am image patch that contains the swc files. The default is None.
    image_patch_shape : list[float], optional
        The xyz dimensions of the bounding box which contains the swc files.
        The default is None.
    img_path : str, optional
        Path to image which is assumed to be stored in a Google Bucket. The
        default is None.
    min_size : int, optional
        Minimum cardinality of swc files that are stored in NeuroGraph. The
        default is the global variable "MIN_SIZE".
    node_spacing : int, optional
        Spacing (in microns) between nodes. The default is the global variable
        "NODE_SPACING".
    progress_bar : bool, optional
        Indication of whether to print out a progress bar during build. The
        default is False.
    prune_connectors : bool, optional
        Indication of whether to prune connectors (see graph_utils.py), sites
        that are likely to be false merges. The default is the global variable
        "PRUNE_CONNECTORS".
    connector_length : int, optional
        Maximum length of connecting paths pruned (see graph_utils.py). The
        default is the global variable "CONNECTOR_LENGTH".
    prune_depth : int, optional
        Branches less than "prune_depth" microns are pruned if "prune" is
        True. The default is the global variable "PRUNE_DEPTH".
    smooth : bool, optional
        Indication of whether to smooth branches from swc files. The default
        is the global variable "SMOOTH".
    swc_dir : str, optional
        Path to a directory containing swc files. The default is None.
    swc_paths : list[str], optional
        List of paths to swc files. The default is None.

    Returns
    -------
    NeuroGraph
        Neurograph generated from swc files stored on local machine.

    """
    # Process swc files
    assert swc_dir or swc_paths, "Provide swc_dir or swc_paths!"
    img_bbox = utils.get_img_bbox(img_patch_origin, img_patch_shape)
    paths = utils.list_paths(swc_dir, ext=".swc") if swc_dir else swc_paths
    swc_dicts, paths = process_local_paths(
        paths, anisotropy=anisotropy, min_size=min_size, img_bbox=img_bbox
    )

    # Filter swc_dicts
    if img_bbox:
        filtered_swc_dicts = []
        for swc_dict in swc_dicts:
            if utils.is_list_contained(img_bbox, swc_dict["xyz"]):
                filtered_swc_dicts.append(swc_dict)
        swc_dicts = filtered_swc_dicts

    # Build neurograph
    neurograph = build_neurograph(
        swc_dicts,
        img_bbox=img_bbox,
        img_path=img_path,
        min_size=min_size,
        node_spacing=node_spacing,
        progress_bar=progress_bar,
        prune_connectors=prune_connectors,
        connector_length=connector_length,
        prune_depth=prune_depth,
        trim_depth=trim_depth,
        smooth=smooth,
        swc_paths=paths,
    )
    return neurograph


def build_neurograph_from_gcs_zips(
    bucket_name,
    gcs_path,
    anisotropy=[1.0, 1.0, 1.0],
    img_path=None,
    min_size=MIN_SIZE,
    node_spacing=NODE_SPACING,
    prune_connectors=PRUNE_CONNECTORS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    trim_depth=TRIM_DEPTH,
    smooth=SMOOTH,
):
    """
    Builds a neurograph from a GCS bucket that contain of zips of swc files.

    Parameters
    ----------
    bucket_name : str
        Name of GCS bucket where zips of swc files are stored.
    gcs_path : str
        Path within GCS bucket to directory containing zips.
    anisotropy : list[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope. The default is [1.0, 1.0, 1.0].
    img_path : str, optional
        Path to image stored GCS Bucket that swc files were generated from.
        The default is None.
    min_size : int, optional
        Minimum cardinality of swc files that are stored in NeuroGraph. The
        default is the global variable "MIN_SIZE".
    node_spacing : int, optional
        Spacing (in microns) between nodes. The default is the global variable
        "NODE_SPACING".
    prune_connectors : bool, optional
        Indication of whether to prune connectors (see graph_utils.py), sites
        that are likely to be false merges. The default is the global variable
        "PRUNE_CONNECTORS".
    connector_length : int, optional
        Maximum length of connecting paths pruned (see graph_utils.py). The
        default is the global variable "CONNECTOR_LENGTH".
    prune_depth : int, optional
        Branches less than "prune_depth" microns are pruned if "prune" is
        True. The default is the global variable "PRUNE_DEPTH".
    smooth : bool, optional
        Indication of whether to smooth branches from swc files. The default
        is the global variable "SMOOTH".

    Returns
    -------
    NeuroGraph
        Neurograph generated from zips of swc files stored in a GCS bucket.

    """
    # Process swc files
    print("Process swc files...")
    total_runtime, t0 = utils.init_timers()
    swc_dicts = download_gcs_zips(bucket_name, gcs_path, min_size, anisotropy)
    t, unit = utils.time_writer(time() - t0)
    print(f"\nModule Runtime: {round(t, 4)} {unit} \n")

    # Build neurograph
    print("Build NeuroGraph...")
    t0 = time()
    neurograph = build_neurograph(
        swc_dicts,
        img_path=img_path,
        min_size=min_size,
        node_spacing=node_spacing,
        prune_connectors=prune_connectors,
        connector_length=connector_length,
        prune_depth=prune_depth,
        trim_depth=trim_depth,
        smooth=smooth,
    )
    t, unit = utils.time_writer(time() - t0)
    print(f"Memory Consumption: {round(utils.get_memory_usage(), 4)} GBs")
    print(f"Module Runtime: {round(t, 4)} {unit} \n")

    return neurograph


# -- Read swc files --
def download_gcs_zips(bucket_name, gcs_path, min_size, anisotropy):
    """
    Downloads swc files from zips stored in a GCS bucket.

    Parameters
    ----------
    bucket_name : str
        Name of GCS bucket where zips are stored.
    gcs_path : str
        Path within GCS bucket to directory containing zips.
    min_size : int
        Minimum cardinality of swc files that are stored in NeuroGraph.
    anisotropy : list[float]
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope.

    Returns
    -------
    swc_dicts : list

    """
    # Initializations
    bucket = storage.Client().bucket(bucket_name)
    zip_paths = utils.list_gcs_filenames(bucket, gcs_path, ".zip")

    # Assign processes
    cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = int(len(zip_paths) * 0.02)
    with ProcessPoolExecutor() as executor:
        processes = []
        print("# zips:", len(zip_paths))
        for i, path in enumerate(zip_paths):
            zip_content = bucket.blob(path).download_as_bytes()
            processes.append(
                executor.submit(
                    process_gcs_zip, zip_content, anisotropy, min_size
                )
            )
            if i > cnt * chunk_size:
                cnt, t1 = utils.report_progress(
                    i + 1, len(zip_paths), chunk_size, cnt, t0, t1
                )

    # Store results
    swc_dicts = []
    for process in as_completed(processes):
        swc_dicts.extend(process.result())
    return swc_dicts


# -- Build neurograph ---
def build_neurograph(
    swc_dicts,
    img_bbox=None,
    img_path=None,
    min_size=MIN_SIZE,
    node_spacing=NODE_SPACING,
    swc_paths=None,
    progress_bar=True,
    prune_connectors=PRUNE_CONNECTORS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    trim_depth=TRIM_DEPTH,
    smooth=SMOOTH,
):
    # Extract irreducibles
    n_components = len(swc_dicts)
    if progress_bar:
        print("# swcs downloaded:", utils.reformat_number(n_components))
    irreducibles, n_nodes, n_edges = get_irreducibles(
        swc_dicts,
        bbox=img_bbox,
        min_size=min_size,
        progress_bar=progress_bar,
        prune_connectors=prune_connectors,
        connector_length=connector_length,
        prune_depth=prune_depth,
        trim_depth=trim_depth,
        smooth=smooth,
    )

    # Build neurograph
    if progress_bar:
        print("\nGraph Overview...")
        print("# connected components:", len(irreducibles))
        print("# nodes:", utils.reformat_number(n_nodes))
        print("# edges:", utils.reformat_number(n_edges))

    neurograph = NeuroGraph(
        img_path=img_path, node_spacing=node_spacing, swc_paths=swc_paths
    )
    while len(irreducibles):
        irreducible_set = irreducibles.pop()
        neurograph.add_component(irreducible_set)
    return neurograph


def get_irreducibles(
    swc_dicts,
    bbox=None,
    min_size=MIN_SIZE,
    progress_bar=True,
    prune_connectors=PRUNE_CONNECTORS,
    connector_length=CONNECTOR_LENGTH,
    prune_depth=PRUNE_DEPTH,
    trim_depth=TRIM_DEPTH,
    smooth=SMOOTH,
):
    n_components = len(swc_dicts)
    chunk_size = int(n_components * 0.02)
    with ProcessPoolExecutor() as executor:
        # Assign Processes
        i = 0
        processes = [None] * n_components
        while swc_dicts:
            swc_dict = swc_dicts.pop()
            processes[i] = executor.submit(
                gutils.get_irreducibles,
                swc_dict,
                bbox,
                prune_connectors,
                connector_length,
                prune_depth,
                trim_depth,
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
                progress_cnt, t1 = utils.report_progress(
                    i + 1, n_components, chunk_size, progress_cnt, t0, t1
                )
    return irreducibles, n_nodes, n_edges


def count_nodes(irreducibles):
    cnt = 0
    for irr_i in irreducibles:
        cnt += len(irr_i["leafs"]) + len(irr_i["junctions"])
    return cnt


def count_edges(irreducibles):
    cnt = 0
    for irr_i in irreducibles:
        cnt += len(irr_i["edges"])
    return cnt
