
"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with SWC files.

An SWC file is a text-based file format used to represent the directed
graphical structure of a neuron. It contains a series of nodes such that each
has the following attributes:
    "id" : node ID
    "type": node type (e.g. soma)
    "x": x coordinate
    "y": y coordinate
    "z": z coordinate
    "pid": node ID of parent

Note: Each uncommented line in an SWC file corresponds to a node and contains
      these attributes in the same order.

"""

from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from google.cloud import storage
from io import BytesIO
from tqdm import tqdm
from zipfile import ZipFile

import networkx as nx
import numpy as np
import os

from deep_neurographs.utils import util


# --- Read ---
class Reader:
    """
    Class that reads SWC files stored in a (1) local directory, (2) local ZIP
    archive, or (3) GCS directory of ZIP archives.

    """

    def __init__(self, anisotropy=[1.0, 1.0, 1.0], min_size=0):
        """
        Initializes a Reader object that loads SWC files.

        Parameters
        ----------
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        min_size : int, optional
            Threshold on the number nodes in SWC files that are parsed and
            returned.

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.min_size = min_size

    def load(self, swc_pointer):
        """
        Loads data from SWC files located at the path specified by
        "swc_pointer".

        Parameters
        ----------
        swc_pointer : dict, list, str
            Object that points to SWC files to be read, must be one of:
                - swc_dir (str): Path to directory containing SWC files.
                - swc_path (str): Path to single SWC file.
                - swc_path_list (List[str]): List of paths to SWC files.
                - swc_zip (str): Path to a ZIP archive containing SWC files.
                - gcs_dict (dict): Dictionary that contains the keys
                  "bucket_name" and "path" to read from a GCS bucket.

        Returns
        -------
        List[dict]
            List of dictionaries whose keys and values are the attribute names
            and values from the SWC files. Each dictionary contains the
            following items:
                - "id": unique identifier of each node in an SWC file.
                - "pid": parent ID of each node.
                - "swc_id": name of swc file.
                - "is_soma": indication of there is a soma node.
                - "radius": radius value corresponding to each node.
                - "xyz": coordinates corresponding to each node.

        """
        if type(swc_pointer) is dict:
            return self.load_from_gcs(swc_pointer)
        if type(swc_pointer) is list:
            return self.load_from_local_paths(swc_pointer)
        if type(swc_pointer) is str:
            if ".zip" in swc_pointer:
                return self.load_from_local_zip(swc_pointer)
            if ".swc" in swc_pointer:
                return self.load_from_local_path(swc_pointer)
            if os.path.isdir(swc_pointer):
                paths = util.list_paths(swc_pointer, extension=".swc")
                return self.load_from_local_paths(paths)
        raise Exception("SWC Pointer is not Valid!")

    # --- Load subroutines ---
    def load_from_local_paths(self, swc_paths):
        """
        Reads a list of SWC files stored on the local machine.

        Paramters
        ---------
        swc_paths : List[str]
            Paths to SWC files stored on the local machine.

        Returns
        -------
        List[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.

        """
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for path in swc_paths:
                processes.append(
                    executor.submit(self.load_from_local_path, path)
                )

            # Store results
            swc_dicts = list()
            for process in as_completed(processes):
                result = process.result()
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def load_from_local_path(self, path):
        """
        Reads a single SWC file stored on the local machine.

        Paramters
        ---------
        path : str
            Path to SWC file stored on the local machine.

        Returns
        -------
        dict
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.

        """
        content = util.read_txt(path)
        if len(content) > self.min_size - 10:
            result = self.parse(content)
            result["swc_id"] = util.get_swc_id(path)
            return result
        else:
            return False

    def load_from_local_zip(self, zip_path):
        """
        Reads SWC files from ZIP archive stored on the local machine.

        Paramters
        ---------
        str : str
            Path to a ZIP archive the local machine.

        Returns
        -------
        List[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.

        """
        with ZipFile(zip_path, "r") as zip_file:
            swc_dicts = list()
            swc_files = [f for f in zip_file.namelist() if f.endswith(".swc")]
            for f in tqdm(swc_files, desc="Read SWCs"):
                result = self.load_from_zipped_file(zip_file, f)
                if result:
                    swc_dicts.append(result)
        return swc_dicts

    def load_from_zipped_file(self, zip_file, path):
        """
        Reads SWC file stored in a ZIP archive.

        Parameters
        ----------
        zip_file : ZipFile
            ZIP archive containing SWC file to be read.
        path : str
            Path to SWC file to be read.

        Returns
        -------
        dict
            Dictionaries whose keys and values are the attribute names and
            values from an SWC file.

        """
        content = util.read_zip(zip_file, path).splitlines()
        if len(content) > self.min_size - 10:
            result = self.parse(content)
            result["swc_id"] = util.get_swc_id(path)
            return result
        else:
            return False

    def load_from_gcs(self, gcs_dict):
        """
        Reads SWC files from ZIP archives stored in a GCS bucket.

        Parameters
        ----------
        gcs_dict : dict
            Dictionary with the keys "bucket_name" and "path".

        Returns
        -------
        List[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.

        """
        # Initializations
        bucket = storage.Client().bucket(gcs_dict["bucket_name"])
        zip_paths = util.list_gcs_filenames(bucket, gcs_dict["path"], ".zip")
        pbar = tqdm(
            total=len(zip_paths), desc="Download SWCs", dynamic_ncols=True
        )

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for path in zip_paths:
                zip_content = bucket.blob(path).download_as_bytes()
                processes.append(
                    executor.submit(self.load_from_cloud_zip, zip_content)
                )

            # Store results
            swc_dicts = list()
            for process in as_completed(processes):
                swc_dicts.extend(process.result())
                pbar.update(1)
        return swc_dicts

    def load_from_cloud_zip(self, zip_content):
        """
        Reads SWC files stored in a ZIP archive downloaded from a cloud
        bucket.

        Parameters
        ----------
        zip_content : ...
            Content of a ZIP archive.

        Returns
        -------
        List[dict]
            List of dictionaries whose keys and values are the attribute
            names and values from an SWC file.


        """
        with ZipFile(BytesIO(zip_content)) as zip_file:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = list()
                for f in util.list_files_in_zip(zip_content):
                    threads.append(
                        executor.submit(
                            self.load_from_zipped_file, zip_file, f
                        )
                    )

                # Process results
                swc_dicts = list()
                for thread in as_completed(threads):
                    result = thread.result()
                    if result:
                        swc_dicts.append(result)
        return swc_dicts

    # --- Process swc content ---
    def parse(self, content):
        """
        Parses an SWC file to extract the content which is stored in a dict.
        Note that node_ids from swc are refactored to index from 0 to n-1
        where n is the number of nodes in the SWC file.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        dict
            Dictionaries whose keys and values are the attribute names
            and values from an SWC file.

        """
        # Initializations
        content, offset = self.process_content(content)
        swc_dict = {
            "id": np.zeros((len(content)), dtype=np.int32),
            "radius": np.zeros((len(content)), dtype=np.float32),
            "pid": np.zeros((len(content)), dtype=np.int32),
            "xyz": np.zeros((len(content), 3), dtype=np.float32),
            "is_soma": False,
        }

        # Parse content
        for i, line in enumerate(content):
            parts = line.split()
            swc_dict["id"][i] = parts[0]
            swc_dict["radius"][i] = float(parts[-2])
            swc_dict["pid"][i] = parts[-1]
            swc_dict["xyz"][i] = self.read_xyz(parts[2:5], offset)
            if int(parts[1]) == 1:
                swc_dict["is_soma"] = True

        # Convert radius from nanometers to microns
        if swc_dict["radius"][0] > 100:
            swc_dict["radius"] /= 1000
        return swc_dict

    def process_content(self, content):
        """
        Processes lines of text from an SWC file, extracting an offset
        value and returning the remaining content starting from the line
        immediately after the last commented line.

        Parameters
        ----------
        content : List[str]
            List of strings such that each is a line from an SWC file.

        Returns
        -------
        List[str]
            A list of strings representing the lines of text starting from the
            line immediately after the last commented line.
        List[float]
            Offset of SWC file.

        """
        offset = [0.0, 0.0, 0.0]
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                offset = self.read_xyz(line.split()[2:5])
            if not line.startswith("#"):
                return content[i:], offset

    def read_xyz(self, xyz_str, offset=[0.0, 0.0, 0.0]):
        """
        Reads a 3D coordinate from a string and transforms it (if applicable).

        Parameters
        ----------
        xyz_str : str
            Coordinate stored as a str.
        offset : List[float], optional
            Offset used to shift coordinates if provided in the SWC file. The
            default is [0.0, 0.0, 0.0].

        Returns
        -------
        numpy.ndarray
            Coordinate of a node from an SWC file.

        """
        xyz = np.zeros((3))
        for i in range(3):
            xyz[i] = self.anisotropy[i] * (float(xyz_str[i]) + offset[i])
        return xyz


# --- Write ---
def write(path, content, color=None):
    """
    Writes content to a specified file in a format based on the type of
    content.

    Parameters
    ----------
    path : str
        File path where the content will be written.
    content : list, dict, nx.Graph
        The content to be written.
    color : str, optional
        Color of swc to be written. The default is None.

    Returns
    -------
    None

    """
    if type(content) is list:
        write_list(path, content, color=color)
    elif type(content) is dict:
        write_dict(path, content, color=color)
    elif type(content) is nx.Graph:
        write_graph(path, content, color=color)
    else:
        raise Exception("Unable to write {} to swc".format(type(content)))


def write_list(path, entry_list, color=None):
    """
    Writes a list of SWC entries to a file at path.

    Parameters
    ----------
    path : str
        Path that swc will be written to.
    entry_list : list[str]
        List of entries that will be written to an swc file.
    color : str, optional
        Color of swc to be written. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        # Preamble
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, x, y, z, r, pid")

        # Entries
        for i, entry in enumerate(entry_list):
            f.write("\n" + entry)


def write_dict(path, swc_dict, color=None):
    """
    Writes the dictionary to an swc file.

    Parameters
    ----------
    path : str
        Path that swc will be written to.
    swc_dict : dict
        Dictionaries whose keys and values are the attribute name and values
        from an swc file.
    color : str, optional
        Color of swc to be written. The default is None.

    Returns
    -------
    None

    """
    graph, _ = to_graph(swc_dict, set_attrs=True)
    write_graph(path, graph, color=color)


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
    List[str]
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


def save_point(path, xyz, radius=5, color=None):
    """
    Writes an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    xyz : ArrayLike
        xyz coordinate to be written to an swc file.
    radius : float, optional
        Radius of point. The default is 5um.
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
            f.write("# id, type, x, y, z, r, pid")
        f.write("\n")

        # Entries
        f.write(make_simple_entry(1, -1, xyz, radius=radius))


def save_edge(path, xyz_1, xyz_2, color=None, radius=5):
    """
    Writes the line segment formed by "xyz_1" and "xyz_2" to an swc file.

    Parameters
    ----------
    path : str
        Path on local machine that swc file will be written to.
    xyz_1 : ArrayLike
        xyz coordinate to be written to an swc file.
    xyz_2 : ArrayLike
        xyz coordinate to be written to an swc file.
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
            f.write("# id, type, x, y, z, r, pid")
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


def make_simple_entry(node, parent, xyz, radius=5):
    """
    Makes an entry to be written in an swc file.

    Parameters
    ----------
    node : int
        Node that entry corresponds to.
    parent : int
         Parent of node "i".
    xyz : numpy.ndarray
        xyz coordinate to be written to an swc file.

    Returns
    -------
    str
        Entry of an swc file

    """
    x, y, z = tuple(xyz)
    return f"{node} 2 {x} {y} {z} {radius} {parent}"


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
    except ValueError:
        radius = 1.0
    return radius


# --- Miscellaneous ---
def to_graph(swc_dict, set_attrs=False):
    """
    Converts an dictionary containing swc attributes to a graph.

    Parameters
    ----------
    swc_dict : dict
        Dictionaries whose keys and values are the attribute name and values
        from an swc file.
    set_attrs : bool, optional
        Indication of whether to set attributes. The default is False.

    Returns
    -------
    networkx.Graph
        Graph generated from "swc_dict".

    """
    graph = nx.Graph(graph_id=swc_dict["swc_id"])
    graph.add_edges_from(zip(swc_dict["id"][1:], swc_dict["pid"][1:]))
    if set_attrs:
        xyz = util.numpy_to_hashable(swc_dict["xyz"])
        xyz_to_node = dict(zip(xyz, swc_dict["id"]))
        __add_attributes(swc_dict, graph)
        return graph, xyz_to_node
    return graph


def __add_attributes(swc_dict, graph):
    """
    Adds node attributes to a NetworkX graph based on information from
    "swc_dict".

    Parameters:
    ----------
    swc_dict : dict
        A dictionary containing SWC data. It must have the following keys:
        - "id": A list of node identifiers (unique for each node).
        - "xyz": A list of 3D coordinates (x, y, z) for each node.
        - "radius": A list of radii for each node.

    graph : networkx.Graph
        A NetworkX graph object to which the attributes will be added.
        The graph must contain nodes that correspond to the IDs in
        "swc_dict["id"]".

    Returns:
    -------
    None

    """
    attrs = dict()
    for idx, node in enumerate(swc_dict["id"]):
        attrs[node] = {
            "xyz": swc_dict["xyz"][idx],
            "radius": swc_dict["radius"][idx],
        }
    nx.set_node_attributes(graph, attrs)
