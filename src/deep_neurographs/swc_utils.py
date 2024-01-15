"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with swc files.

"""

from io import BytesIO
import networkx as nx
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from zipfile import ZipFile

from deep_neurographs import geometry_utils
from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils


# -- io utils --
def process_local_paths(paths, min_size, bbox=None):
    swc_dicts = dict()
    for path in paths:
        swc_id, swc_dict = parse_local_swc(path, bbox=bbox, min_size=min_size)
        if len(swc_dict["id"]) > min_size:
            swc_dicts[swc_id] = swc_dict
    return swc_dicts


def process_gsc_zip(bucket, zip_path, min_size=0):
    swc_dicts = dict()
    zip_blob = bucket.blob(zip_path)
    zip_content = zip_blob.download_as_bytes()
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            threads = [
                executor.submit(parse_gcs_zip, zip_file, path, min_size)
                for path in utils.list_files_in_gcs_zip(zip_content)
            ]
        for thread in as_completed(threads):
            swc_id, result = thread.result()
            if len(result["id"]) > min_size:
                swc_dicts[swc_id] = result
    return swc_dicts


def parse_local_swc(path, bbox=None, min_size=0):
    contents = read_from_local(path)
    parse_bool = len(contents) > min_size
    if parse_bool:
        swc_dict = parse(contents, bbox=bbox) if bbox else fast_parse(contents)
    else:
        swc_dict = {"id": [-1]}
    return utils.get_swc_id(path), swc_dict


def parse_gcs_zip(zip_file, path, min_size=0):
    contents = read_from_gcs_zip(zip_file, path)
    swc_dict = fast_parse(contents) if len(contents) > min_size else {"id": [-1]}
    return utils.get_swc_id(path), swc_dict


def parse(contents, bbox=None):
    """
    Parses an swc file to extract the contents which is stored in a dict. Note
    that node_ids from swc are refactored to index from 0 to n-1 where n is
    the number of entries in the swc file.

    Parameters
    ----------
    path : str
        Path to an swc file.
    ...

    Returns
    -------
    ...

    """
    min_id = np.inf
    offset = [0, 0, 0]
    swc_dict = {"id": [], "radius": [], "pid": [], "xyz": []}
    contents, swc_id = get_contents(contents)
    for line in contents:
        parts = line.split()
        xyz = read_xyz(parts[2:5], offset=offset)
        if bbox:
            if not utils.is_contained(bbox, xyz):
                break
        swc_dict["id"].append(int(parts[0]))
        swc_dict["radius"].append(float(parts[-2]))
        swc_dict["pid"].append(int(parts[-1]))
        swc_dict["xyz"].append(xyz)
        if swc_dict["id"][-1] < min_id:
            min_id = swc_dict["id"][-1]

    # Reindex from zero
    for i in range(len(swc_dict["id"])):
        swc_dict["id"][i] -= min_id
        swc_dict["pid"][i] -= min_id

    return swc_dict if len(swc_dict["id"]) > 1 else {"id": [-1]}


def fast_parse(contents):
    """
    Parses an swc file to extract the contents which is stored in a dict. Note
    that node_ids from swc are refactored to index from 0 to n-1 where n is
    the number of entries in the swc file.

    Parameters
    ----------
    path : str
        Path to an swc file.
    ...

    Returns
    -------
    ...

    """
    contents, offset = get_contents(contents)
    dtype = np.int16 if len(contents) < 2 ** 16 else np.int32
    swc_dict = {
        "id": np.zeros((len(contents)), dtype=dtype),
        "radius": np.zeros((len(contents)), dtype=np.float16),
        "pid": np.zeros((len(contents)), dtype=dtype),
        "xyz": []
    }

    min_id = np.inf
    for i, line in enumerate(contents):
        parts = line.split()
        xyz = read_xyz(parts[2:5], offset=offset)
        swc_dict["id"][i] = int(parts[0])
        swc_dict["radius"][i] = float(parts[-2])
        swc_dict["pid"][i] = int(parts[-1])
        swc_dict["xyz"].append(xyz)

    # Reindex from zero
    min_id = np.min(swc_dict["id"])
    swc_dict["id"] -= min_id
    swc_dict["pid"] -= min_id
    return swc_dict


def get_contents(swc_contents):
    offset = [0, 0, 0]
    for i, line in enumerate(swc_contents):
        if line.startswith("# OFFSET"):
            parts = line.split()
            offset = read_xyz(parts[2:5])
        if not line.startswith("#"):
            break
    return swc_contents[i:], offset


def read_from_local(path):
    """
    Reads swc file stored at "path" on local machine.

    Parameters
    ----------
    Path : str
        Path to swc file to be read.

    Returns
    -------
    list
        List such that each entry is a line from the swc file.

    """
    with open(path, "r") as file:
        return file.readlines()


def read_from_gcs_zip(zip_file, path):
    """
    Reads the content of an swc file from a zip file in a GCS bucket.

    """
    with zip_file.open(path) as text_file:
        return text_file.read().decode("utf-8").splitlines()


def read_xyz(xyz, offset=[0, 0, 0]):
    """
    Reads the (z,y,x) coordinates from an swc file, then reverses and scales
    them.

    Parameters
    ----------
    zyx : str
        (z,y,x) coordinates.

    Returns
    -------
    tuple
        The (x,y,z) coordinates from an swc file.

    """
    return tuple([np.float32(xyz[i]) + offset[i] for i in range(3)])


def write(path, contents):
    if type(contents) is list:
        write_list(path, contents)
    elif type(contents) is dict:
        write_dict(path, contents)
    elif type(contents) is nx.Graph:
        write_graph(path, contents)
    else:
        assert True, "Unable to write {} to swc".format(type(contents))


def write_list(path, entry_list, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path that swc will be written to.
    entry_list : list[list[int]]
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
        for i, entry in enumerate(entry_list):
            for x in entry:
                f.write(str(x) + " ")
            f.write("\n")


def write_dict(path, swc_dict, color=None):
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


def write_graph(path, graph):
    """
    Makes a list of entries to be written in an swc file.

    Parameters
    ----------
    path : str
        Path that swc will be written to.
    graph : networkx.Graph
        Graph to be written to swc file.

    Returns
    -------
    list[str]
        List of swc file entries to be written.

    """
    # loop through connected components

    reindex = dict()
    for i, j in graph.edges:
        if len(reindex) < 1:
            entry, reindex = make_entry(graph, i, -1, reindex)
            entry_list = [entry]
        entry, reindex = make_entry(graph, j, reindex[i], reindex)
        entry_list.append(entry)
    return entry_list


def make_entry(graph, i, parent, r, reindex):
    """
    Makes an entry to be written in an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that "i" and "parent" belong to.
    i : int
        Node that entry corresponds to.
    parent : int
         Parent of node "i".
    r : float
        Radius.
    reindex : dict
        Converts 'graph node id' to 'swc node id'.

    """
    reindex[i] = len(reindex) + 1
    r = graph.nodes[i]["radius"]
    x, y, z = tuple(map(str, graph.nodes[i]["xyz"]))
    return [x, y, z, r, parent], reindex


# -- Conversions --
def to_graph(swc_dict, graph_id=None, set_attrs=False):
    graph = nx.Graph(graph_id=graph_id)
    graph.add_edges_from(zip(swc_dict["id"][1:], swc_dict["pid"][1:]))
    if set_attrs:
        graph = __add_attributes(swc_dict, graph)
        xyz_to_node = dict(zip(swc_dict["xyz"], swc_dict["id"]))
        return graph, xyz_to_node
    return graph


def __add_attributes(swc_dict, graph):
    xyz = swc_dict["xyz"]
    radii = swc_dict["radius"]
    attrs = [{"xyz": xyz[i], "radius": radii[i]} for i in graph.nodes]
    nx.set_node_attributes(graph, dict(zip(swc_dict["id"], attrs)))
    return graph


# -- miscellaneous --
def smooth(swc_dict):
    if len(swc_dict["xyz"]) > 10:
        xyz = np.array(swc_dict["xyz"])
        graph = to_graph(swc_dict)
        leafs, junctions = gutils.get_irreducible_nodes(graph)
        if len(junctions) == 0:
            xyz = geometry_utils.smooth_branch(xyz)
        else:
            idxs = []
            root = None
            for i, j in nx.dfs_edges(graph, source=leafs[0]):
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
    xyz[idxs] = geometry_utils.smooth_branch(xyz[idxs], s=10)
    return xyz