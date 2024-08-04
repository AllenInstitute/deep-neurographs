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

from deep_neurographs import geometry, utils


# -- io utils --
def process_local_paths(
    paths, anisotropy=[1.0, 1.0, 1.0], min_size=5, img_bbox=None
):
    """
    Iterates over a list of swc paths to swc file, then builds a dictionary
    where the keys are swc attributes (i.e. id, xyz, radius, pid) and values
    are the corresponding contents within the swc file.

    Parameters
    ----------
    paths : list[str]
        List of paths to swc files to be parsed.
    min_size : int, optional
        Threshold on the number of nodes contained in an swc file. Only swc
        files with more than "min_size" nodes are stored in "swc_dicts". The
        default is 3.
    img_bbox : dict, optional
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image. Only swc files with at least one node contained in
        "img_bbox" are stored in "swc_dicts". The default is None.

    Returns
    -------
    swc_dicts : list
        List of dictionaries where the keys are swc attributes (i.e. id, xyz,
        radius, pid) and values are the corresponding contents within the swc
        file.

    """
    valid_paths = []
    swc_dicts = []
    for path in paths:
        # Read contents
        contents = read_from_local(path)
        if len(contents) > min_size:
            swc_dict = parse(contents, anisotropy=anisotropy)
            swc_dict["swc_id"] = utils.get_swc_id(path)
            swc_dicts.append(swc_dict)
            valid_paths.append(path)
    return swc_dicts, valid_paths


def process_gcs_zip(zip_content, anisotropy=[1.0, 1.0, 1.0], min_size=0):
    swc_dicts = []
    with ZipFile(BytesIO(zip_content)) as zip_file:
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = [
                executor.submit(
                    parse_gcs_zip, zip_file, path, anisotropy, min_size
                )
                for path in utils.list_files_in_gcs_zip(zip_content)
            ]

            # Process results
            for thread in as_completed(threads):
                result = thread.result()
                if len(result["id"]) > 0:
                    swc_dicts.append(result)
    return swc_dicts


def parse_gcs_zip(zip_file, path, anisotropy=[1.0, 1.0, 1.0], min_size=0):
    # Parse contents
    contents = read_from_gcs_zip(zip_file, path)
    if len(contents) > min_size:
        swc_dict = parse(contents, anisotropy=anisotropy)
    else:
        swc_dict = {"id": []}

    # Store id
    swc_id = utils.get_swc_id(path)
    swc_dict["swc_id"] = swc_id
    return swc_dict


def parse(contents, anisotropy=[1.0, 1.0, 1.0]):
    """
    Parses an swc file to extract the contents which is stored in a dict. Note
    that node_ids from swc are refactored to index from 0 to n-1 where n is
    the number of entries in the swc file.

    Parameters
    ----------
    contents : list[str]
        List of entries from an swc file.
    anisotropy : list[float]

    Returns
    -------
    ...

    """
    # Compile swc content
    contents, offset = get_contents(contents)
    swc_dict = {
        "id": np.zeros((len(contents)), dtype=np.int32),
        "radius": np.zeros((len(contents)), dtype=np.float32),
        "pid": np.zeros((len(contents)), dtype=np.int32),
        "xyz": np.zeros((len(contents), 3), dtype=np.float32),
    }
    for i, line in enumerate(contents):
        parts = line.split()
        swc_dict["id"][i] = parts[0]
        swc_dict["radius"][i] = float(parts[-2])
        swc_dict["pid"][i] = parts[-1]
        swc_dict["xyz"][i] = read_xyz(
            parts[2:5], anisotropy=anisotropy, offset=offset
        )

    # Check whether radius is in nanometers
    if swc_dict["radius"][0] > 100:
        swc_dict["radius"] /= 1000
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
    try:
        with zip_file.open(path) as txt_file:
            return txt_file.read().decode("utf-8").splitlines()
    except:
        print(f"Failed to read {path}")
        return []


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
    entry_list : list[str]
        List of entries that will be written to an swc file.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        # Preamble
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, z, y, x, r, pid")

        # Entries
        for i, entry in enumerate(entry_list):
            f.write("\n" + entry)


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
    node_to_idx = {-1: -1}
    for i, j in nx.dfs_edges(graph):
        # Initialize entry list
        if len(node_to_idx) == 1:
            entry, node_to_idx = make_entry(graph, i, -1, node_to_idx)
            entry_list = [entry]

        # Add entry
        entry, node_to_idx = make_entry(graph, j, i, node_to_idx)
        entry_list.append(entry)
    write_list(path, entry_list)


def save_point(path, xyz, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    entry_list : list[list]
        List of entries that will be written to an swc file.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        # Preamble
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")

        # Entries
        f.write(make_simple_entry(1, -1, xyz))


def save_edge(path, xyz_1, xyz_2, color=None, radius=6):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    entry_list : list[list]
        List of entries to be written to an swc file.
    color : str, optional
        Color of nodes. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        # Preamble
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, z, y, x, r, pid")
        f.write("\n")

        # Entries
        f.write(make_simple_entry(1, -1, xyz_1, radius=radius))
        f.write("\n")
        f.write(make_simple_entry(2, 1, xyz_2, radius=radius))


def make_entry(graph, i, parent, node_to_idx):
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
    node_to_idx : dict
        Converts 'graph node id' to 'swc node id'.

    Returns
    -------
    ...

    """
    r = graph.nodes[i]["radius"]
    x, y, z = tuple(graph.nodes[i]["xyz"])
    node_to_idx[i] = len(node_to_idx)
    entry = f"{node_to_idx[i]} 2 {x} {y} {z} {r} {node_to_idx[parent]}"
    return entry, node_to_idx


def set_radius(graph, i):
    """
    Sets the radius of node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph containing node "i".
    i : int
        Node.

    Returns
    -------
    float
        Radius of node "i".
    """
    try:
        radius = graph[i]["radius"]
    except:
        radius = 1.0
    return radius


def make_simple_entry(node, parent, xyz, radius=8):
    """
    Makes an entry to be written in an swc file.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that "i" and "parent" belong to.
    node : int
        Node that entry corresponds to.
    parent : int
         Parent of node "i".
    anisotropy : list[float]
        Image to real-world coordinates scaling factors for (x, y, z) that is
        applied to swc files.

    """
    x, y, z = tuple(xyz)
    return f"{node} 2 {x} {y} {z} {radius} {parent}"


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
