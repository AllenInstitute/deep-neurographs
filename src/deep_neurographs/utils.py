"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General routines for various tasks.

"""


import json
import os
import shutil
import zarr

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.linalg import svd


# --- dictionary utils ---
def remove_item(my_set, item):
    try:
        my_set.remove(item)
    except:
        pass
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
    try:
        return my_dict[key]
    except:
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
def read_n5(path):
    """
    Reads n5 file at "path".

    Parameters
    ----------
    path : str
        Path to n5.

    Returns
    -------
    np.array
        Image volume.
    """
    return zarr.open(zarr.N5FSStore(path), "r").volume

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
def dist(x, y, metric="l2"):
    """
    Computes distance between "x" and "y".

    Parameters
    ----------

    Returns
    -------
    float

    """
    if metric == "l1":
        return np.linalg.norm(np.subtract(x, y), ord=1)
    else:
        return np.linalg.norm(np.subtract(x, y), ord=2)


def pair_dist(pair_1, pair_2, metric="l2"):
    pair_1.reverse()
    d1 = _pair_dist(pair_1, pair_2)

    pair_1.reverse()
    d2 = _pair_dist(pair_1, pair_2)
    return min(d1, d2)


def _pair_dist(pair_1, pair_2, metric="l2"):
    d1 = dist(pair_1[0], pair_2[0], metric=metric)
    d2 = dist(pair_1[1], pair_2[1], metric=metric)
    return max(d1, d2)


def smooth_branch(xyz, round=True):
    t = np.arange(len(xyz[:, 0]) + 12)
    s = len(t) / 10
    cs_x = UnivariateSpline(t, extend_boundary(xyz[:, 0]), s=s, k=3)
    cs_y = UnivariateSpline(t, extend_boundary(xyz[:, 1]), s=s, k=3)
    cs_z = UnivariateSpline(t, extend_boundary(xyz[:, 2]), s=s, k=3)
    smoothed_x = trim_boundary(cs_x(t))
    smoothed_y = trim_boundary(cs_y(t))
    smoothed_z = trim_boundary(cs_z(t))
    smoothed = np.column_stack((smoothed_x, smoothed_y, smoothed_z))
    if round:
        return smoothed #np.round(smoothed).astype(int)
    else:
        return smoothed


def extend_boundary(x, num_boundary_points=6):
    extended_x = np.concatenate((
        np.linspace(x[0], x[1], num_boundary_points, endpoint=False),
        x,
        np.linspace(x[-2], x[-1], num_boundary_points, endpoint=False)
    ))
    return extended_x


def trim_boundary(x, num_boundary_points=6):
    return x[num_boundary_points:-num_boundary_points]


def check_img_path(target_labels, xyz_1, xyz_2):
    d = dist(xyz_1, xyz_2)
    t_steps = np.arange(0, 1, 1 / d)
    num_steps = len(t_steps)
    labels = set()
    collisions = set()
    for t in t_steps:
        xyz = tuple([int(line(xyz_1[i], xyz_2[i], t)) for i in range(3)])
        if target_labels[xyz] != 0:
            # Check for repeat collisions
            if xyz in collisions:
                num_steps -= 1
            else:
                collisions.add(xyz)

            # Check for collision with multiple labels
            labels.add(target_labels[xyz])
            if len(labels) > 1:
                return False
    ratio = len(collisions) / len(t_steps)
    return True if ratio > 1 / 3 else False
        
    
    
def line(xyz_1, xyz_2, t):
    return np.round((1 - t) * xyz_1 + t * xyz_2)


def to_world(xyz, anisotropy, shift=[0, 0, 0]):
    return tuple([int((xyz[i] - shift[i]) * anisotropy[i]) for i in range(3)])


def to_img(xyz, anisotropy, shift=[0, 0, 0]):
    return tuple([int((xyz[i] - shift[i]) / anisotropy[i]) for i in range(3)])


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
