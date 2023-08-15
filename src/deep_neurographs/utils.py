"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General routines for various tasks.

"""


import json
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- dictionary utils ---
def remove_item(my_set, item):
    """ """
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
def mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


def listdir(path, ext=None):
    if ext is None:
        return [f for f in os.listdir(path)]
    else:
        return [f for f in os.listdir(path) if ext in f]


def listsubdirs(path, keyword=None):
    subdirs = []
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            if keyword is None:
                subdirs.append(d)
            elif keyword in d:
                subdirs.append(d)
    return subdirs


# --- io utils ---
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


# --- plot utils ---
def plot(data, title):
    fig = go.Figure(data=data)
    fig.update_layout(
        plot_bgcolor="white",
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    fig.update_layout(
        scene=dict(aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.8)),
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

    fig.update_layout(width=1200, height=600)
    fig.show()


# --- miscellaneous ---
def dist(x, y):
    """
    Computes Euclidean distance between "x" and "y".

    Parameters
    ----------

    Returns
    -------
    float

    """
    return np.linalg.norm(np.subtract(x, y))


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
