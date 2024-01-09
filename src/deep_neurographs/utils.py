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

import numpy as np
import tensorstore as ts
import zarr

ANISOTROPY = [0.748, 0.748, 1.0]
SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "zarr"]


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
    ts_arr = ts.open(
        {
            "driver": driver,
            "kvstore": {
                "driver": "gcs",
                "bucket": "allen-nd-goog",
                "path": path,
            },
        }
    ).result()
    if driver == "neuroglancer_precomputed":
        return ts_arr[ts.d["channel"][0]]
    elif driver == "zarr":
        ts_arr = ts_arr[0, 0, :, :, :]
        ts_arr = ts_arr[ts.d[0].transpose[2]]
        ts_arr = ts_arr[ts.d[0].transpose[1]]
        return ts_arr


def read_img_chunk(img, xyz, shape):
    start, end = get_start_end(xyz, shape)
    return img[
        start[2]:end[2], start[1]:end[1], start[0]:end[0]
    ].transpose(2, 1, 0)


def get_chunk(arr, xyz, shape):
    start, end = get_start_end(xyz, shape)
    return deepcopy(
        arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    )


def read_tensorstore(ts_arr, xyz, shape):
    start, end = get_start_end(xyz, shape)
    return (
        ts_arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        .read()
        .result()
    )


def get_start_end(xyz, shape):
    start = [xyz[i] - shape[i] // 2 for i in range(3)]
    end = [xyz[i] + shape[i] // 2 for i in range(3)]
    return start, end


def get_superchunks(img_path, label_path, xyz, shape, from_center=True):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        img_job = executor.submit(
            get_superchunk,
            img_path,
            "zarr",
            xyz,
            shape,
            from_center=from_center,
        )
        label_job = executor.submit(
            get_superchunk,
            label_path,
            "neuroglancer_precomputed",
            xyz,
            shape,
            from_center=from_center,
        )
    img = img_job.result().astype(np.int16)
    label = label_job.result().astype(np.int64)
    return img, label


def get_superchunk(path, driver, xyz, shape, from_center=True):
    ts_arr = open_tensorstore(path, driver)
    if from_center:
        return read_tensorstore(ts_arr, xyz, shape)
    else:
        xyz = [xyz[i] + shape[i] // 2 for i in range(3)]
        return read_tensorstore(ts_arr, xyz, shape)


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


# --- coordinate conversions ---
def world_to_img(neurograph, node_or_xyz):
    if type(node_or_xyz) == int:
        node_or_xyz = deepcopy(neurograph.nodes[node_or_xyz]["xyz"])
    return to_img(node_or_xyz, shift=neurograph.origin)


def img_to_patch(xyz, patch_centroid, patch_dims):
    half_patch_dims = [patch_dims[i] // 2 for i in range(3)]
    return np.round(xyz - patch_centroid + half_patch_dims).astype(int)


def patch_to_img(xyz, patch_centroid, patch_dims):
    half_patch_dims = [patch_dims[i] // 2 for i in range(3)]
    return np.round(xyz + patch_centroid - half_patch_dims).astype(int)


def to_world(xyz, shift=[0, 0, 0]):
    return tuple([xyz[i] * ANISOTROPY[i] - shift[i] for i in range(3)])


def to_img(xyz, shift=[0, 0, 0]):
    return apply_anisotropy(xyz - shift, return_int=True)


def apply_anisotropy(xyz, return_int=False):
    if return_int:
        return [round(xyz[i] / ANISOTROPY[i]) for i in range(3)]
    else:
        return [xyz[i] / ANISOTROPY[i] for i in range(3)]


# --- math utils ---
def get_avg_std(data, weights=None):
    avg = np.average(data, weights=weights)
    var = np.average((data - avg) ** 2, weights=weights)
    return avg, math.sqrt(var)


def is_contained(bbox, img_shape, xyz):
    xyz = apply_anisotropy(xyz - bbox["min"])
    for i in range(3):
        lower_bool = xyz[i] < 0
        upper_bool = xyz[i] >= img_shape[i]
        if lower_bool or upper_bool:
            return False
    return True


# --- miscellaneous ---
def get_id(path):
    """
    Gets segment id of the swc file at "path".

    """
    filename = path.split("/")[-1]
    return filename.replace(".0.swc", "")


def get_img_mip(img, axis=0):
    return np.max(img, axis=axis)


def normalize_img(img):
    img -= np.min(img)
    return img / np.max(img)


def reformat_number(number):
    return f"{number:,}"


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


def progress_bar(current, total, bar_length=50):
    progress = int(current / total * bar_length)
    bar = (
        f"[{'=' * progress}{' ' * (bar_length - progress)}] {current}/{total}"
    )
    print(f"\r{bar}", end="", flush=True)

def xor(a, b):
    if (a and b) or (not a and not b):
        return False
    else:
        return True
    