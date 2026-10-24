"""
Created on Sun July 16 14:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from random import sample

import psutil


# --- Dictionary Utils ---
def find_best(my_dict, maximize=True):
    """
    Finds the key associated with the largest integer or longest list.

    Parameters
    ----------
    my_dict : dict
        Dictionary to be searched.
    maximize : bool, optional
        Indication of whether to find the largest/longest or
        smallest/shortest.

    Returns
    -------
    hashable
        Key associated with the longest list or largest integer in "my_dict".
    """

    def score(v):
        """
        Assigns a score to a given value.
        """
        return v if isinstance(v, (int, float)) else len(v)

    optimize = max if maximize else min
    return optimize(my_dict, key=lambda k: score(my_dict[k]))


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
    return {k: v for k, v in my_dict.items() if k not in keys}


# --- Miscellaneous ---
def get_memory_usage():
    """
    Gets the current memory usage in gigabytes.

    Returns
    -------
    float
        Current memory usage in gigabytes.
    """
    return psutil.virtual_memory().used / 1e9


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


def sample_once(my_container):
    """
    Samples a single element from the given container.

    Parameters
    ----------
    my_container : container
        Container to be sampled from.

    Returns
    -------
    hashable
        Single element from the given container.
    """
    if not isinstance(my_container, list):
        return sample(list(my_container), 1)[0]
    else:
        return sample(my_container, 1)[0]


def time_writer(t, unit="seconds"):
    """
    Converts a runtime to a larger unit of time if applicable.

    Parameters
    ----------
    t : float
        Runtime.
    unit : str, optional
        Unit that the given time is expressed in. Default is "seconds".

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
