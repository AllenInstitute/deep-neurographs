"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with swc files.

"""

import os
import random
from copy import deepcopy as cp

import networkx as nx
import numpy as np
from more_itertools import zip_broadcast
from deep_neurographs import graph_utils as gutils, utils


def read_swc(path):
    with open(path, "r") as file:
        contents = file.readlines()
        file.close()
    return contents


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
    swc_dict = {"id": [], "xyz": [], "radius": [], "pid": []}

    # Parse raw data
    min_id = np.inf
    offset = [0.0, 0.0, 0.0]
    for line in raw_swc:
        if line.startswith("# OFFSET"):
            parts = line.split()
            offset = read_xyz(parts[2:5])
        if not line.startswith("#") and len(line) > 0:
            parts = line.split()
            swc_dict["id"].append(int(parts[0]))
            swc_dict["radius"].append(float(parts[-2]))
            swc_dict["pid"].append(int(parts[-1]))
            swc_dict["xyz"].append(
                read_xyz(parts[2:5], anisotropy=anisotropy, offset=offset)
            )
            if swc_dict["id"][-1] < min_id:
                min_id = swc_dict["id"][-1]

    # Reindex from zero
    for i in range(len(swc_dict["id"])):
        swc_dict["id"][i] -= min_id
        swc_dict["pid"][i] -= min_id

    return swc_dict


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0]):
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
    xyz = [int(float(xyz[i]) * anisotropy[i] + offset[i]) for i in range(3)]
    return tuple(xyz)


def write_swc(path, list_of_entries, color=None):
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
    with open(path, "w") as f:
        if color is not None:
            f.write("# COLOR" + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")
        for i, entry in enumerate(list_of_entries):
            for x in entry:
                f.write(str(x) + " ")
            f.write("\n")


def write_swc_dict(path, swc_dict, color=None):
    with open(path, "w") as f:
        if color is not None:
            f.write("# COLOR" + color)
        else:
            f.write("# id, type, x, y, z, r, pid")
        f.write("\n")
        shift = 1 if swc_dict["id"][0] == 0 else 0
        first = True
        for i in swc_dict["id"]:
            f.write(str(i + shift) + " " + str(2) + " ")
            for j in range(3):
                f.write(str(swc_dict["xyz"][i][j]) + " ")
            pid = -1 if first else swc_dict["pid"][i] + shift
            f.write(str(swc_dict["radius"][i]) + " ")
            f.write(str(pid) + " ")
            f.write("\n")
            first = False


def file_to_graph(swc_dict, graph_id=None, set_attrs=False):
    graph = nx.Graph(graph_id=graph_id)
    graph.add_edges_from(zip(swc_dict["id"][1:], swc_dict["pid"][1:]))
    if set_attrs:
        for i in graph.nodes:
            graph.nodes[i]["xyz"] = swc_dict["xyz"][i]
            graph.nodes[i]["radius"] = swc_dict["radius"][i]
    return graph


def dir_to_graphs(swc_dir):
    list_of_graphs = []
    for f in utils.listdir(swc_dir, ext=".swc"):
        swc_dict = parse(read_swc(os.path.join(swc_dir, f)))
        graph = file_to_graph(swc_dict, graph_id=f, set_attrs=True)
        list_of_graphs.append(graph)
    return list_of_graphs


def file_to_volume(swc_dict, sparse=False, vid=None, radius_plus=0):
    volume = []
    for i in swc_dict["id"]:
        r = max(3 * int(np.round(swc_dict["radius"][i] + radius_plus)), 5)
        xyz = cp(swc_dict["xyz"][i])
        volume.extend(generate_coords(xyz, r))
    return dict(zip_broadcast(volume, vid)) if sparse else np.array(volume)


def dir_to_volume(swc_dir, radius_plus=0):
    volume = dict()
    for vid, f in enumerate(utils.listdir(swc_dir, ext=".swc")):
        swc_dict = smooth(parse(read_swc(os.path.join(swc_dir, f))))
        volume.update(
            file_to_volume(
                swc_dict, sparse=True, vid=f, radius_plus=radius_plus
            )
        )
    return volume


def smooth(swc_dict):
    if len(swc_dict["xyz"]) > 10:
        xyz = np.array(swc_dict["xyz"], dtype=int)
        graph = file_to_graph(swc_dict)
        leafs, junctions =  gutils.get_irreducibles(graph)
        if len(junctions) == 0:
            xyz = utils.smooth_branch(xyz)
        else:
            idxs = []
            root = None
            for (i, j) in nx.dfs_edges(graph, source=leafs[0]):
                # Check start of path is valid
                if root is None:
                    root = i
                    idxs = [i]                

                # Check whether to end path
                idxs.append(j)
                if j in leafs + junctions:
                    root = None
                    if len(idxs) > 10:
                        xyz = upd_edge(xyz, idxs)
        swc_dict["xyz"] = [tuple(xyz_i) for xyz_i in xyz]
    return swc_dict


def upd_edge(xyz, idxs):
    idxs = np.array(idxs)
    xyz[idxs] = utils.smooth_branch(xyz[idxs].copy())
    return xyz

def upd_dict(upd, path, radius, pid):
    for k in range(len(path)):
        next_id = len(upd["id"])
        upd["id"].append(next_id)
        upd["xyz"].append(path[k])
        upd["radius"].append(radius[k])
        if len(upd["pid"]) == 0:
            upd["pid"].extend([1, 0])
        elif k == 0:
            upd["pid"].append(pid)
        else:
            upd["pid"].append(next_id - 1)
    return upd, next_id


def generate_coords(center, r):
    xyz = []
    for x in range(-r, r + 1):
        for y in range(-r, r + 1):
            for z in range(-r, r + 1):
                if abs(x) + abs(y) + abs(z) <= r:
                    xyz.append((center[0] + x, center[1] + y, center[2] + z))
    return xyz
