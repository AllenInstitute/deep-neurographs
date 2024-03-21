"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General helper routines for various tasks.

"""

import concurrent.futures
import json
import math
import os
import shutil
from copy import deepcopy
from io import BytesIO
from random import sample
from time import time
from zipfile import ZipFile

import numpy as np
import psutil
import tensorstore as ts
import zarr
from skimage.color import label2rgb

ANISOTROPY = np.array([0.748, 0.748, 1.0])
SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "n5", "zarr"]


# --- dictionary utils ---
def remove_item(my_set, item):
    """
    Removes item from a set.

    Parameters
    ----------
    my_set : set
        Set to be queried.
    item :
        Value to query.

    Returns
    -------
        Set "my_set" with "item" removed if it existed.

    """
    if item in my_set:
        my_set.remove(item)
    return my_set


def check_key(my_dict, key):
    """
    Checks whether "key" is contained in "my_dict". If so, returns the
    corresponding value.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be checked
    key : hashable data type

    Returns
    -------
    dict value or bool
       If "key" is a key in "my_dict", then the associated value is returned.
       Otherwise, the bool "False" is returned.

    """
    if key in my_dict.keys():
        return my_dict[key]
    else:
        return False


def remove_key(my_dict, key):
    """
    Removes "key" from "my_dict" in the case when key may need to be reversed.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be queried
    key : hashable data type
        Key to query.

    Returns
    -------
    my_dict : dict
        Dictionary "my_dict" with key-value associated with "key" removed if
        it existed.

    """
    if check_key(my_dict, key):
        my_dict.pop(key)
    elif check_key(my_dict, (key[1], key[0])):
        my_dict.pop((key[1], key[0]))
    return my_dict


def append_dict_value(my_dict, key, value):
    """
    Appends "value" to the list stored at "key".

    Parameters
    ----------
    my_dict : dict
        Dictionary to be queried.
    key : hashable data type
        Key to be query.
    value : list item type
        Value to append to list stored at "key".

    Returns
    -------
    my_dict : dict
        Updated dictionary.

    """
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


# --- os utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at "path".

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. The default is False.

    Returns
    -------
    None

    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes directory and all subdirectories at "path".

    Parameters
    ----------
    path : str
        Path to directory and subdirectories to be deleted if they exist.

    Returns
    -------
    None

    """
    if os.path.exists(path):
        shutil.rmtree(path)


def listdir(path, ext=None):
    """
    Lists all files in the directory at "path". If an extension "ext" is
    provided, then only files containing "ext" are returned.

    Parameters
    ----------
    path : str
        Path to directory to be searched.

    ext : str, optional
       Extension of file type of interest. The default is None.

    Returns
    -------
    list
        List of all files in directory at "path" with extension "ext" if
        provided. Otherwise, list of all files in directory.

    """
    if ext is None:
        return [f for f in os.listdir(path)]
    else:
        return [f for f in os.listdir(path) if ext in f]


def list_subdirs(path, keyword=None):
    """
    Creates list of all subdirectories at "path". If "keyword" is provided,
    then only subdirectories containing "keyword" are contained in list.

    Parameters
    ----------
    path : str
        Path to directory containing subdirectories to be listed.

    keyword : str, optional
        Only subdirectories containing "keyword" are contained in list that is
        returned. The default is None.

    Returns
    -------
    list
        List of all subdirectories at "path".

    """
    subdirs = []
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            if keyword is None:
                subdirs.append(d)
            elif keyword in d:
                subdirs.append(d)
    return subdirs


def list_paths(directory, ext=None):
    """
    Lists all paths within "directory".

    Parameters
    ----------
    directory : str
        Directory to be searched.
    ext : str, optional
        If provided, only paths of files with the extension "ext" are
        returned. The default is None.

    Returns
    -------
    list[str]
        List of all paths within "directory".

    """
    paths = []
    for f in listdir(directory, ext=ext):
        paths.append(os.path.join(directory, f))
    return paths


# -- gcs utils --
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


# --- io utils ---
def open_zarr(path):
    """
    Opens zarr file at "path".

    Parameters
    ----------
    path : str
        Path to zarr file to be opened.

    Returns
    -------
    np.ndarray
        Contents of zarr file at "path".

    """
    n5store = zarr.N5FSStore(path, "r")
    if "653980" in path:
        return zarr.open(n5store).ch488.s0
    elif "653158" in path:
        return zarr.open(n5store).s0


def open_tensorstore(path, driver):
    """
    Uploads segmentation mask stored as a directory of shard files.

    Parameters
    ----------
    path : str
        Path to directory containing shard files.
    driver : str
        Storage driver needed to read data at "path".

    Returns
    -------
    sparse_volume : dict
        Sparse image volume.

    """
    assert driver in SUPPORTED_DRIVERS, "Error! Driver is not supported!"
    arr = ts.open(
        {
            "driver": driver,
            "kvstore": {
                "driver": "gcs",
                "bucket": "allen-nd-goog",
                "path": path,
            },
            "context": {
                "cache_pool": {"total_bytes_limit": 1000000000},
                "cache_pool#remote": {"total_bytes_limit": 1000000000},
                "data_copy_concurrency": {"limit": 8},
            },
            "recheck_cached_data": "open",
        }
    ).result()
    if driver == "neuroglancer_precomputed":
        return arr[ts.d["channel"][0]]
    return arr


def read_tensorstore(arr, xyz, shape, from_center=True):
    chunk = get_chunk(arr, xyz, shape, from_center=from_center)
    return chunk.read().result()


def get_chunk(arr, xyz, shape, from_center=True):
    start, end = get_start_end(xyz, shape, from_center=from_center)
    return deepcopy(
        arr[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
    )


def get_start_end(xyz, shape, from_center=True):
    if from_center:
        start = [xyz[i] - shape[i] // 2 for i in range(3)]
        end = [xyz[i] + shape[i] // 2 for i in range(3)]
    else:
        start = xyz
        end = [xyz[i] + shape[i] for i in range(3)]
    return start, end


def get_superchunks(img_path, labels_path, xyz, shape, from_center=True):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        img_job = executor.submit(
            get_superchunk,
            img_path,
            "n5" if ".n5" in img_path else "zarr",
            xyz,
            shape,
            from_center=from_center,
        )
        labels_job = executor.submit(
            get_superchunk,
            labels_path,
            "neuroglancer_precomputed",
            xyz,
            shape,
            from_center=from_center,
        )
    img = img_job.result().astype(np.int16)
    labels = labels_job.result().astype(np.int64)
    assert img.shape == labels.shape, "img.shape != labels.shape"
    return img, labels


def get_superchunk(path, driver, xyz, shape, from_center=True):
    arr = open_tensorstore(path, driver)
    return read_tensorstore(arr, xyz, shape, from_center=from_center)


def read_json(path):
    """
    Reads json file stored at "path".

    Parameters
    ----------
    path : str
        Path where json file is stored.

    Returns
    -------
    dict
        Contents of json file.

    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_txt(path):
    """
    Reads txt file stored at "path".

    Parameters
    ----------
    path : str
        Path where txt file is stored.

    Returns
    -------
    str
        Contents of txt file.

    """
    with open(path, "r") as f:
        return f.read()


def parse_metadata(path):
    metadata = read_json(path)
    return metadata["chunk_origin"], metadata["chunk_shape"]


def write_json(path, contents):
    """
    Writes "contents" to a .json file at "path".

    Parameters
    ----------
    path : str
        Path that .txt file is written to.
    contents : dict
        Contents to be written to .txt file.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        json.dump(contents, f)


def write_txt(path, contents):
    f = open(path, "w")
    f.write(contents)
    f.close()


# --- coordinate conversions ---
def img_to_patch(xyz, patch_centroid, patch_dims):
    half_patch_dims = [patch_dims[i] // 2 for i in range(3)]
    patch_coord = xyz - patch_centroid + half_patch_dims
    return tuple(patch_coord.astype(int))


def patch_to_img(xyz, patch_centroid, patch_dims):
    half_patch_dims = [patch_dims[i] // 2 for i in range(3)]
    return np.round(xyz + patch_centroid - half_patch_dims).astype(int)


def to_world(xyz, shift=[0, 0, 0]):
    return tuple([xyz[i] * ANISOTROPY[i] - shift[i] for i in range(3)])


def to_img(xyz):
    return (xyz / ANISOTROPY).astype(int)


# --- math utils ---
def get_avg_std(data, weights=None):
    avg = np.average(data, weights=weights)
    var = np.average((data - avg) ** 2, weights=weights)
    return avg, math.sqrt(var)


def is_contained(bbox, xyz, buffer=0):
    above = any(xyz > bbox["max"] - buffer)
    below = any(xyz < bbox["min"] + buffer)
    return False if above or below else True


def is_list_contained(bbox, xyz_list):
    return any([is_contained(bbox, to_img(xyz)) for xyz in xyz_list])


def sample_singleton(my_container):
    return sample(my_container, 1)[0]


# --- miscellaneous ---
def get_img_bbox(origin, shape):
    """
    Origin is assumed to be top, front, left corner.

    """
    if origin and shape:
        origin = np.array(origin)
        shape = np.array(shape)
        return {"min": origin, "max": origin + shape}
    else:
        return None


def get_swc_id(path):
    """
    Gets segment id of the swc file at "path".

    """
    filename = path.split("/")[-1]
    return filename.split(".")[0]


def get_img_mip(img, axis=0):
    return np.max(img, axis=axis)


def get_labels_mip(img, axis=0):
    mip = np.max(img, axis=axis)
    mip = label2rgb(mip)
    return (255 * mip).astype(np.uint8)


def normalize_img(img):
    img -= np.min(img)
    return img / np.max(img)


def numpy_to_hashable(arr):
    return [tuple(item) for item in arr.tolist()]


def reformat_number(number):
    return f"{number:,}"


def get_memory_usage():
    return psutil.virtual_memory().used / 1e9


def init_timers():
    return time(), time()


def progress_bar(current, total, bar_length=50, eta=None, runtime=None):
    progress = int(current / total * bar_length)
    n_completed = f"Completed: {current}/{total}"
    bar = f"[{'=' * progress}{' ' * (bar_length - progress)}]"
    eta = f"Time Remaining: {eta}" if eta else ""
    runtime = f"Estimated Total Runtime: {runtime}" if runtime else ""
    print(f"\r{bar} {n_completed} | {eta} | {runtime}    ", end="", flush=True)


def time_writer(t, unit="seconds"):
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit


def find_best(my_dict):
    best_key = None
    best_vote_cnt = 0
    if len(my_dict) > 0:
        for key, values in my_dict.items():
            vote_cnt = len(values)
            if vote_cnt > best_vote_cnt:
                best_key = key
                best_vote_cnt = vote_cnt
    return best_key, best_vote_cnt
