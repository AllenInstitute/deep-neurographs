"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with swc files.

"""


import numpy as np


def parse(raw_swc, anisotropy=[1.0, 1.0, 1.0]):
    """
    Parses a raw swc file to extract the (x,y,z) coordinates and radii. Note
    that node_ids from swc are refactored to index from 0 to n-1 where n is
    the number of entries in the swc file.

    Parameters
    ----------
    raw_swc : list[str]
        Contents of an swc file.
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for [x, y, z] due to
        anistropy of the microscope.

    Returns
    -------
    dict
        The (x,y,z) coordinates and radii stored in "raw_swc".

    """
    # Initialize swc
    swc_dict = {
        "id": [],
        "xyz": [],
        "radius": [],
        "pid": [],
    }

    # Parse raw data
    min_id = np.inf
    for line in raw_swc:
        if not line.startswith("#") and len(line) > 0:
            parts = line.split()
            swc_dict["id"].append(int(parts[0]))
            swc_dict["xyz"].append(read_xyz(parts[2:5], anisotropy=anisotropy))
            swc_dict["radius"].append(float(parts[-2]))
            swc_dict["pid"].append(int(parts[-1]))
            if swc_dict["id"][-1] < min_id:
                min_id = swc_dict["id"][-1]

    # Reindex from zero
    for i in range(len(swc_dict["id"])):
        swc_dict["id"][i] -= min_id
        swc_dict["pid"][i] -= min_id
    return swc_dict


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0]):
    """
    Reads the (z,y,x) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    zyx : str
        (z,y,x) coordinates.
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for [x, y, z] due to
        anistropy of the microscope.

    Returns
    -------
    list
        The (x,y,z) coordinates from an swc file.

    """
    return tuple([int(float(xyz[i]) * anisotropy[i]) for i in range(3)])


def write_swc(path_to_swc, list_of_entries, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path_to_swc : str
        Path that swc will be written to.
    list_of_entries : list[list[int]]
        List of entries that will be written to an swc file.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path_to_swc, "w") as f:
        if color is not None:
            f.write("# COLOR" + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")
        for i, entry in enumerate(list_of_entries):
            for x in entry:
                f.write(str(x) + " ")
            f.write("\n")
