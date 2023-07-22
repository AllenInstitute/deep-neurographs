"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

General routines for various tasks.

"""


import os

import numpy as np


def remove_item(my_set, item):
    """
    """
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


def mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


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
