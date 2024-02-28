"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with swc files.

"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from zipfile import ZipFile

import networkx as nx
import numpy as np

from deep_neurographs import geometry
from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils


# -- io utils --
def process_local_paths(paths, anisotropy=[1.0, 1.0, 1.0], min_size=0, img_bbox=None):
    """
    Iterates over a list of paths to swc files and calls a routine that builds
    a dictionary where the keys are swc attributes (i.e. id, xyz, radius, pid)
    and values are the corresponding contents within the swc file.

    Parameters
    ----------
    paths : list[str]
        List of paths to swc files to be parsed.
    min_size : int, optional
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "swc_dicts". The
        default is 0.
    img_bbox : dict, optional
        Dictionary with the keys "min" and "max" which specify a bounding box
        in the image. Only swc files with at least one node contained in
        "img_bbox" are stored in "swc_dicts". The default is None.

    Returns
    -------
    swc_dicts : list
        List of dictionaries where the keys are swc attributes (i.e. id, xyz,
        radius, pid) and values are the corresponding contents within the swc
        file.

    """
    swc_dicts = []
    for path in paths:
        swc_dict = parse_local_swc(
            path, anisotropy=anisotropy, img_bbox=img_bbox, min_size=min_size
        )
        if len(swc_dict["id"]) > min_size:
            swc_dicts.append(swc_dict)
    return swc_dicts


def process_gsc_zip(bucket, zip_path, anisotropy=[1.0, 1.0, 1.0], min_size=0):
    swc_dicts = []
    zip_blob = bucket.blob(zip_path)
    zip_content = zip_blob.download_as_bytes()
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            threads = [
                executor.submit(
                    parse_gcs_zip, zip_file, path, anisotropy, min_size
                )
                for path in utils.list_files_in_gcs_zip(zip_content)
            ]
        for thread in as_completed(threads):
            result = thread.result()
            if len(result["id"]) > min_size:
                swc_dicts.append(result)
    return swc_dicts


def parse_local_swc(path, anisotropy=[1.0, 1.0, 1.0], img_bbox=None, min_size=0):
    # Parse contents
    contents = read_from_local(path)
    parse_bool = len(contents) > min_size
    if parse_bool and img_bbox:
        swc_dict = parse(contents, img_bbox, anisotropy=anisotropy)
    elif parse_bool:
        swc_dict = fast_parse(contents, anisotropy=anisotropy)
    else:
        swc_dict = {"id": [-1]}

    # Store id
    swc_id = utils.get_swc_id(path)
    swc_dict["swc_id"] = swc_id
    return swc_dict


def parse_gcs_zip(zip_file, path, anisotropy=[1.0, 1.0, 1.0], min_size=0):
    # Parse contents
    contents = read_from_gcs_zip(zip_file, path)
    parse_bool = len(contents) > min_size
    if parse_bool:
        swc_dict = fast_parse(contents, anisotropy=anisotropy)
    else:
        swc_dict = {"id": [-1]}

    # Store id
    swc_id = utils.get_swc_id(path)
    swc_dict["swc_id"] = swc_id
    return swc_dict


def parse(contents, img_bbox, anisotropy=[1.0, 1.0, 1.0]):
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
    contents, offset = get_contents(contents)
    swc_dict = {"id": [], "radius": [], "pid": [], "xyz": []}
    for line in contents:
        parts = line.split()
        xyz = read_xyz(parts[2:5], anisotropy=anisotropy, offset=offset)
        if img_bbox:
            img_coord = utils.to_img(np.array(xyz))
            if not utils.is_contained(img_bbox, img_coord, buffer=8):
                break
        swc_dict["id"].append(int(parts[0]))
        swc_dict["radius"].append(float(parts[-2]))
        swc_dict["pid"].append(int(parts[-1]))
        swc_dict["xyz"].append(xyz)
        if swc_dict["id"][-1] < min_id:
            min_id = swc_dict["id"][-1]

    # Reindex from zero and reformat
    if len(swc_dict["id"]) > 0:
        swc_dict["id"] = np.array(swc_dict["id"], dtype=int) - min_id
        swc_dict["pid"] = np.array(swc_dict["pid"], dtype=int) - min_id
        swc_dict["radius"] = np.array(swc_dict["radius"])
        swc_dict["xyz"] = np.array(swc_dict["xyz"])

    return swc_dict if len(swc_dict["id"]) > 1 else {"id": [-1]}


def fast_parse(contents, anisotropy=[1.0, 1.0, 1.0]):
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
    min_id = np.inf
    swc_dict = {
        "id": np.zeros((len(contents)), dtype=np.int32),
        "radius": np.zeros((len(contents)), dtype=np.float32),
        "pid": np.zeros((len(contents)), dtype=np.int32),
        "xyz": np.zeros((len(contents), 3), dtype=np.float32),
    }
    for i, line in enumerate(contents):
        parts = line.split()
        swc_dict["id"][i] = parts[0]
        swc_dict["radius"][i] = parts[-2]
        swc_dict["pid"][i] = parts[-1]
        swc_dict["xyz"][i] = read_xyz(
            parts[2:5], anisotropy=anisotropy, offset=offset
        )

    # Reindex from zero
    min_id = np.min(swc_dict["id"])
    swc_dict["id"] -= min_id
    swc_dict["pid"] -= min_id
    swc_dict["radius"] /= 1000.0
    return swc_dict


def reindex(arr, idxs):
    return arr[idxs]


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


def read_xyz(xyz, anisotropy=[1.0, 1.0, 1.0], offset=[0, 0, 0]):
    """
    Reads the (x,y,z)) coordinates from an swc file, then shift and scales
    them if application.

    Parameters
    ----------
    xyz : str
        (z,y,x) coordinates.

    Returns
    -------
    tuple
        The (x,y,z) coordinates from an swc file.

    """
    xyz = [float(xyz[i]) + offset[i] for i in range(3)]
    return tuple([xyz[i] * anisotropy[i] for i in range(3)])


def write(path, contents, color=None):
    if type(contents) is list:
        write_list(path, contents, color=color)
    elif type(contents) is dict:
        write_dict(path, contents, color=color)
    elif type(contents) is nx.Graph:
        write_graph(path, contents, color=color)
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
            for item in entry:
                f.write(str(item) + " ")
            f.write("\n")


def write_dict(path, swc_dict, color=None):
    graph, _ = to_graph(swc_dict, set_attrs=True)
    return write_graph(path, graph, color=color)


def write_graph(path, graph, color=None):
    """
    Makes a list of entries to be written in an swc file. This routine assumes
    that "graph" has a single connected components.

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
    reindex = dict()
    for i, j in nx.dfs_edges(graph):
        # Initialize entry list
        if len(reindex) < 1:
            r = set_radius(graph, i)
            entry, reindex = make_entry(graph, i, -1, r, reindex)
            entry_list = [entry]

        # Add entry
        r = set_radius(graph, j)
        entry, reindex = make_entry(graph, j, reindex[i], r, reindex)
        entry_list.append(entry)
    write_list(path, entry_list)


def set_radius(graph, i):
    try:
        return graph[i]["radius"]
    except:
        return 2


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
    return [reindex[i], 2, x, y, z, r, parent], reindex


# -- Conversions --
def to_graph(swc_dict, graph_id=None, set_attrs=False):
    graph = nx.Graph(graph_id=graph_id)
    graph.add_edges_from(zip(swc_dict["id"][1:], swc_dict["pid"][1:]))
    if set_attrs:
        xyz = swc_dict["xyz"]
        if type(swc_dict["xyz"]) == np.ndarray:
            xyz = utils.numpy_to_hashable(swc_dict["xyz"])
        graph = __add_attributes(swc_dict, graph)
        xyz_to_node = dict(zip(xyz, swc_dict["id"]))
        return graph, xyz_to_node
    return graph


def __add_attributes(swc_dict, graph):
    attrs = dict()
    for idx, node_id in enumerate(swc_dict["id"]):
        attrs[node_id] = {
            "xyz": swc_dict["xyz"][idx],
            "radius": swc_dict["radius"][idx],
        }
    nx.set_node_attributes(graph, attrs)
    return graph


# -- miscellaneous --
def upd_edge(xyz, idxs):
    idxs = np.array(idxs)
    xyz[idxs] = geometry.smooth_branch(xyz[idxs], s=10)
    return xyz
