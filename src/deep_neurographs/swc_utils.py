"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with swc files.

"""

import os
import numpy as np
import networkx as nx
import random
from copy import deepcopy as cp
from deep_neurographs import utils
from more_itertools import zip_broadcast
from scipy.signal import savgol_filter


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


def file_to_volume(swc_dict, sparse=False, vid=None):
    volume = []
    for i in swc_dict["id"]:
        r = int(np.round(swc_dict["radius"][i]))
        xyz = cp(swc_dict["xyz"][i])
        volume.extend(generate_coords(xyz, r))
    return dict(zip_broadcast(volume, vid)) if sparse else np.array(volume)


def dir_to_volume(swc_dir):
    volume = dict()
    for vid, f in enumerate(utils.listdir(swc_dir, ext=".swc")):
        swc_dict = parse(read_swc(os.path.join(swc_dir, f)))
        volume.update(file_to_volume(swc_dict, sparse=True, vid=vid+1))
    return volume

def resample_swc(swc_dict):
    graph = file_to_graph(swc_dict, set_attrs=True)
    root = random.sample(graph.nodes, 1)[0]
    old_to_new = {root: 0}
    upd = {
        "id": [0],
        "xyz": [graph.nodes[root]["xyz"]],
        "radius": [graph.nodes[root]["radius"]],
        "pid": [],
    }
    for (i, j) in nx.bfs_edges(graph, source=root):
        # Extract info
        xyz_1 = np.array(graph.nodes[i]["xyz"])
        xyz_2 = np.array(graph.nodes[j]["xyz"])
        r_1 = graph.nodes[i]["radius"]
        r_2 = graph.nodes[j]["radius"]

        # Generate path between nodes
        if np.sum(abs(xyz_1 - xyz_2)) > 2:
            xyz = run_randomwalk(xyz_1, xyz_2)
            r = np.mean([r_1, r_2]) * np.ones(len(xyz))
        else:
            xyz = [xyz_1, xyz_2]
            r = [r_1, r_2]
        upd, new_id_j = upd_dict(upd, xyz, r, old_to_new[i])
        old_to_new[j] = new_id_j
    return upd


def run_randomwalk(start, end):
    walk = []
    pointer = start.copy()
    idx = 0
    while (pointer != end).any():
        #idx = random.sample([0, 1, 2], 1)
        idx = (idx + 1) % 3
        if pointer[idx] != end[idx]:
            pointer[idx] += np.sign(end[idx] - pointer[idx])
            walk.append(pointer.copy())
    return moving_average(walk, 4)


def moving_average(data, window):
    data = np.array(data)
    weights = np.ones(window) / window
    padded_data = np.pad(
        data, ((window - 1, window - 1), (0, 0)), mode='edge',
    )
    smoothed_data = np.apply_along_axis(
        lambda x: np.convolve(x, weights, mode='valid'), axis=0, arr=padded_data
    )
    return smoothed_data.astype(np.int)


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
                if x**2 + y**2 + z**2 <= r**2:
                    xyz.append((center[0] + x, center[1] + y, center[2] + z))
    return xyz
