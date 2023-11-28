"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General routines for various tasks.

"""

import concurrent.futures
import json
import os
import shutil

import numpy as np
import plotly.graph_objects as go
import tensorstore as ts
import zarr
from plotly.subplots import make_subplots

ANISOTROPY = [0.748, 0.748, 1.0]
SUPPORTED_DRIVERS = ["neuroglancer_precomputed", "zarr"]


# --- dictionary utils ---
def remove_item(my_set, item):
    if item in my_set:
        my_set.remove(item)
    return my_set


def check_key(my_dict, key):
    """
    Checks whether "key" is contained in dictionary. If so, returns the
    corresponding value.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be checked
    key : all

    Returns
    -------
    dict value or bool

    """
    if key in my_dict.keys():
        return my_dict[key]
    else:
        return False


def remove_key(my_dict, key):
    """
    Removes key from dict in the case when key may need to be reversed

    """
    if check_key(my_dict, key):
        my_dict.pop(key)
    elif check_key(my_dict, (key[1], key[0])):
        my_dict.pop((key[1], key[0]))
    else:
        print("key not found")
    return my_dict


# --- os utils ---
def mkdir(path, delete=False):
    if os.path.exists(path) and delete:
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def listdir(path, ext=None):
    if ext is None:
        return [f for f in os.listdir(path)]
    else:
        return [f for f in os.listdir(path) if ext in f]


def list_subdirs(path, keyword=None):
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
    return img[
        xyz[2] - shape[2] // 2: xyz[2] + shape[2] // 2,
        xyz[1] - shape[1] // 2: xyz[1] + shape[1] // 2,
        xyz[0] - shape[0] // 2: xyz[0] + shape[0] // 2,
    ].transpose(2, 1, 0)


def get_chunk(arr, xyz, shape):
    xyz_1 = [max(xyz[i] - shape[i] // 2, 0) for i in range(3)]
    xyz_2 = [min(xyz[i] + shape[i] // 2, arr.shape[i] - 1) for i in range(3)]
    return arr[xyz_1[0] : xyz_2[0], xyz_1[1] : xyz_2[1], xyz_1[2] : xyz_2[2]]


def read_tensorstore(ts_arr, xyz, shape):
    return (
        ts_arr[
            xyz[0] - shape[0] // 2: xyz[0] + shape[0] // 2,
            xyz[1] - shape[1] // 2: xyz[1] + shape[1] // 2,
            xyz[2] - shape[2] // 2: xyz[2] + shape[2] // 2,
        ]
        .read()
        .result()
    )


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
    return img_job.result(), label_job.result()


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

    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_txt(path):
    with open(path, "r") as f:
        return f.read()


def read_mistake_coords(path, anisotropy=[1.0, 1.0, 1.0], shift=[0, 0, 0]):
    xyz = []
    with open(path, "r") as file:
        for line in file:
            if not line.startswith("#") and len(line) > 0:
                parts = line.split()
                xyz_1 = extract_coords(parts[0:3])
                xyz_2 = extract_coords(parts[3:6])
                xyz.append(to_img(xyz_1, anisotropy, shift=shift))
                xyz.append(to_img(xyz_2, anisotropy, shift=shift))
    return np.array(xyz)


def read_mistake_log(path):
    splits_log = dict()
    with open(path, "r") as file:
        for line in file:
            if not line.startswith("#") and len(line) > 0:
                parts = line.split()
                xyz_1 = extract_coords(parts[0:3])
                xyz_2 = extract_coords(parts[3:6])
                swc_1 = parts[6].replace(",", "")
                swc_2 = parts[7].replace(",", "")
                key = frozenset([swc_1, swc_2])
                splits_log[key] = {
                    "swc": [swc_1, swc_2],
                    "xyz": [xyz_1, xyz_2],
                }
    return splits_log


def extract_coords(parts):
    coords = []
    for p in parts:
        p = p.replace("[", "").replace("]", "").replace(",", "")
        coords.append(float(p))
    return np.array(coords, dtype=int)


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


# --- plot utils ---
def plot(data, title):
    fig = go.Figure(data=data)
    fig.update_layout(
        plot_bgcolor="white",
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    fig.update_layout(
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=1)),
        width=1200,
        height=600,
    )
    fig.show()


def subplot(data1, data2, title):
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
    )
    fig.add_trace(data1, row=1, col=1)
    fig.add_trace(data2, row=1, col=2)
    fig.update_layout(title_text=title, showlegend=True)

    fig.update_xaxes(row=1, col=1, matches="y")
    fig.update_yaxes(row=1, col=1, matches="x")
    fig.update_layout(
        scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1)
    )

    # Update the size of the second subplot
    fig.update_xaxes(row=1, col=2, matches="y")
    fig.update_yaxes(row=1, col=2, matches="x")
    fig.update_layout(
        scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1)
    )

    fig.update_layout(width=1200, height=800)
    fig.show()


# --- miscellaneous ---
def normalize_img(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img


def to_world(xyz, shift=[0, 0, 0]):
    return tuple([int((xyz[i] - shift[i]) * ANISOTROPY[i]) for i in range(3)])


def to_img(xyz, shift=[0, 0, 0]):
    xyz = apply_anisotropy(xyz - shift, return_int=True)
    return tuple(xyz)


def apply_anisotropy(xyz, return_int=False):
    if return_int:
        return [int(xyz[i] / ANISOTROPY[i]) for i in range(3)]
    else:
        return [xyz[i] / ANISOTROPY[i] for i in range(3)]


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
