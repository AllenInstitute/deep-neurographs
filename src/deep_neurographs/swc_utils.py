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
    min_id = np.inf
    nRows = len(raw_swc)
    swc_dict = {
        "subnodes": nRows * [None],
        "xyz": nRows * [None],
        "radius": nRows * [None],
        "parents": nRows * [None],
    }
    for i, line in enumerate(raw_swc):
        parts = line.split()
        swc_dict["subnodes"][i] = int(parts[0])
        swc_dict["xyz"][i] = read_xyz(parts[2:5], anisotropy=anisotropy)
        swc_dict["radius"][i] = float(parts[-2])
        swc_dict["parents"][i] = int(parts[-1])
        if swc_dict["subnodes"][i] < min_id:
            min_id = swc_dict["subnodes"][i]
        swc_dict["subnodes"][i] -= min_id
        swc_dict["parents"][i] -= min_id
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
    xyz = list(map(float, xyz))
    xyz.reverse()
    return tuple([xyz[i] / anisotropy[i] for i in range(3)])


def extract_topo_nodes(nodes, parents):
    parents = set(parents)
    nodes = set(nodes)
    leafs = nodes.difference(parents)
    # add junctions
    return leafs, None
