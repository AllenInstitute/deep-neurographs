"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General helper routines for various tasks.

"""

import json
import math
import os
import shutil
from io import BytesIO
from random import sample
from time import time
from zipfile import ZipFile

import numpy as np
import psutil

from deep_neurographs import img_utils


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
    set
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
    dict
        Updated dictionary.

    """
    if check_key(my_dict, key):
        my_dict.pop(key)
    elif check_key(my_dict, (key[1], key[0])):
        my_dict.pop((key[1], key[0]))
    return my_dict


def remove_items(my_dict, keys):
    """
    Removes dictionary items corresponding to "keys".

    Parameters
    ----------
    my_dict : dict
        Dictionary to be edited.
    keys : list
        List of keys to be deleted from "my_dict".

    Returns
    -------
    dict
        Updated dictionary.

    """
    for key in keys:
        if key in my_dict.keys():
            del my_dict[key]
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
    dict
        Updated dictionary.

    """
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


def find_best(my_dict, maximize=True):
    """
    Given a dictionary where each value is either a list or int (i.e. cnt),
    finds the key associated with the longest list or largest integer.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be searched.
    maximize : bool, optional
        Indication of whether to find the largest value or highest vote cnt.

    Returns
    -------
    hashable data type
        Key associated with the longest list or largest integer in "my_dict".

    """
    best_key = None
    best_vote_cnt = 0 if maximize else np.inf
    for key in my_dict.keys():
        val_type = type(my_dict[key])
        vote_cnt = my_dict[key] if val_type == float else len(my_dict[key])
        if vote_cnt > best_vote_cnt and maximize:
            best_key = key
            best_vote_cnt = vote_cnt
        elif vote_cnt < best_vote_cnt and not maximize:
            best_key = key
            best_vote_cnt = vote_cnt
    return best_key


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
        Files in directory at "path" with extension "ext" if provided.
        Otherwise, list of all files in directory.

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


def set_path(dir_name, filename, ext):
    """
    Sets the path for a file in a directory. If a file with the same name
    exists, then this routine finds a suffix to append to the filename.

    Parameters
    ----------
    dir_name : str
        Name of directory that path will be generated to point to.
    filename : str
        Name of file that path will contain.
    ext : str
        Extension of file.

    Returns
    -------
    str
        Path to file in "dirname" with the name "filename" and possibly some
        suffix.

    """
    cnt = 0
    ext = ext.replace(".", "")
    path = os.path.join(dir_name, f"{filename}.{ext}")
    while os.path.exists(path):
        path = os.path.join(dir_name, f"{filename}.{cnt}.{ext}")
        cnt += 1
    return path


# -- gcs utils --
def list_files_in_gcs_zip(zip_content):
    """
    Lists all files in a zip file stored in a GCS bucket.

    Parameters
    ----------
    zip_content : str
        Content stored in a zip file in the form of a string of bytes.

    Returns
    -------
    list[str]
        List of filenames in a zip file.

    """
    with ZipFile(BytesIO(zip_content), "r") as zip_file:
        return zip_file.namelist()


def list_gcs_filenames(bucket, cloud_path, extension):
    """
    Lists all files in a GCS bucket with the given extension.

    Parameters
    ----------
    bucket : google.cloud.client
        Name of bucket to be read from.
    cloud_path : str
        Path to directory in "bucket".
    extension : str
        File extension of filenames to be listed.

    Returns
    -------
    list
        Filenames stored at "cloud" path with the given extension.

    """
    blobs = bucket.list_blobs(prefix=cloud_path)
    return [blob.name for blob in blobs if extension in blob.name]


# -- io utils --
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
        return json.load(f)


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


def parse_metadata(path, anisotropy=[1.0, 1.0, 1.0]):
    """
    Parses metadata file to extract the "chunk_origin" and "chunk_shape".

    Parameters
    ----------
    path : str
        Path to metadata file to be read.
    anisotropy : list[float], optional
        Anisotropy to be applied to values of interest that converts
        coordinates from voxels to world. The default is [1.0, 1.0, 1.0].

    Returns
    -------
    list, list
        Chunk origin and chunk shape specified by metadata.

    """
    metadata = read_json(path)
    origin = metadata["chunk_origin"]
    chunk_origin = img_utils.to_voxels(origin, anisotropy=anisotropy)
    return chunk_origin.tolist(), metadata["chunk_shape"]


def write_json(path, contents):
    """
    Writes "contents" to a json file at "path".

    Parameters
    ----------
    path : str
        Path that .txt file is written to.
    contents : dict
        Contents to be written to json file.

    Returns
    -------
    None

    """
    with open(path, "w") as f:
        json.dump(contents, f)


def write_txt(path, contents):
    """
    Writes "contents" to a txt file at "path".

    Parameters
    ----------
    path : str
        Path that txt file is written to.
    contents : dict
        Contents to be written to txt file.

    Returns
    -------
    None

    """
    f = open(path, "w")
    f.write(contents)
    f.close()


# --- math utils ---
def get_avg_std(data, weights=None):
    """
    Computes the average and standard deviation of "data". If "weights" is
    provided, the weighted average and standard deviation are computed.

    Parameters
    ----------
    data : numpy.ndarray
        Data to be evaluated.
    weights : numpy.ndarray, optional
        Weights to apply to each point in "data". The default is None.

    Returns
    -------
    float, float
        Average and standard deviation of "data".

    """
    avg = np.average(data, weights=weights)
    var = np.average((data - avg) ** 2, weights=weights)
    return avg, math.sqrt(var)


def is_contained(bbox, voxel):
    """
    Checks whether "xyz" is contained within "bbox".

    """
    above = any(voxel >= bbox["max"])
    below = any(voxel < bbox["min"])
    return False if above or below else True


def is_list_contained(bbox, voxels):
    """
    Checks whether every element in "xyz_list" is contained in "bbox".

    Parameters
    ----------
    bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in the image.
    voxels
        List of xyz coordinates to be checked.

    Returns
    -------
    bool
        Indication of whether every element in "voxels" is contained in
        "bbox".

    """
    return all([is_contained(bbox, voxel) for voxel in voxels])


def sample_singleton(my_container):
    """
    Samples a single element from "my_container".

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    sample

    """
    return sample(my_container, 1)[0]


# --- runtime ---
def init_timers():
    """
    Initializes two timers.

    Parameters
    ----------
    None

    Returns
    -------
    time.time
        Timer.
    time.time
        Timer.

    """
    return time(), time()


def time_writer(t, unit="seconds"):
    """
    Converts a runtime "t" to a larger unit of time if applicable.

    Parameters
    ----------
    t : float
        Runtime.
    unit : str, optional
        Unit of time that "t" is expressed in.

    Returns
    -------
    float
        Runtime
    str
        Unit of time.

    """
    assert unit in ["seconds", "minutes", "hours"]
    upd_unit = {"seconds": "minutes", "minutes": "hours"}
    if t < 60 or unit == "hours":
        return t, unit
    else:
        t /= 60
        unit = upd_unit[unit]
        t, unit = time_writer(t, unit=unit)
    return t, unit


def progress_bar(current, total, bar_length=50, eta=None, runtime=None):
    progress = int(current / total * bar_length)
    n_completed = f"Completed: {current}/{total}"
    bar = f"[{'=' * progress}{' ' * (bar_length - progress)}]"
    eta = f"Time Remaining: {eta}" if eta else ""
    runtime = f"Estimated Total Runtime: {runtime}" if runtime else ""
    print(f"\r{bar} {n_completed} | {eta} | {runtime}    ", end="", flush=True)


def report_progress(current, total, chunk_size, cnt, t0, t1):
    eta = get_eta(current, total, chunk_size, t1)
    runtime = get_runtime(current, total, chunk_size, t0, t1)
    progress_bar(current, total, eta=eta, runtime=runtime)
    return cnt + 1, time()


def get_eta(current, total, chunk_size, t0, return_str=True):
    chunk_runtime = time() - t0
    remaining = total - current
    eta = remaining * (chunk_runtime / max(chunk_size, 1))
    t, unit = time_writer(eta)
    return f"{round(t, 4)} {unit}" if return_str else eta


def get_runtime(current, total, chunk_size, t0, t1):
    eta = get_eta(current, total, chunk_size, t1, return_str=False)
    total_runtime = time() - t0 + eta
    t, unit = time_writer(total_runtime)
    return f"{round(t, 4)} {unit}"


# --- miscellaneous ---
def get_swc_id(path):
    """
    Gets segment id of the swc file at "path".

    Parameters
    ----------
    path : str
        Path to swc file.

    Returns
    -------
    str
        Segment id.

    """
    filename = path.split("/")[-1]
    return filename.split(".")[0]


def numpy_to_hashable(arr):
    """
    Converts a numpy array to a hashable data structure.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    list
        Hashable items from "arr".

    """
    return [tuple(item) for item in arr.tolist()]


def reformat_number(number):
    """
    Reformats large number to have commas.

    Parameters
    ----------
    number : float
        Number to be reformatted.

    Returns
    -------
    str
        Reformatted number.

    """
    return f"{number:,}"


def get_memory_usage():
    """
    Gets the current memory usage.

    Parameters
    ----------
    None

    Returns
    -------
    float
        Current memory usage.

    """
    return psutil.virtual_memory().used / 1e9
