"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for reading and writing swc files.

"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from io import BytesIO
from zipfile import ZipFile

import networkx as nx
import numpy as np
from google.cloud import storage
from tqdm import tqdm

from deep_neurographs.utils import util


# --- Read ---
class Reader:
    """
    Class that reads swc files that are stored as (1) local directory of swcs,
    (2) gcs directory of zips containing swcs, (3) local zip containing swcs,
    (4) list of local paths to swcs, or (5) single path to a local swc.

    """

    def __init__(self, anisotropy=[1.0, 1.0, 1.0], min_size=0):
        """
        Initializes a Reader object that loads swc files.

        Parameters
        ----------
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        min_size : int, optional
            Threshold on the number of nodes in swc file. Only swc files with
            more than "min_size" nodes are processed. The default is 0.

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.min_size = min_size

    def load(self, swc_pointer):
        """
        Loads swc files specififed by "swc_pointer" and builds an attributed
        graphs from them.

        Parameters
        ----------
        swc_pointer : dict, list, str
            Object that points to swc files to be read, see class documentation
            for details.

        Returns
        -------
        List[networkx.Graph] or networkx.Graph
            Attributed graphs.

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

    def load_from_local_paths(self, path_list):
        """
        Reads swc files from local machine and builds an attributed graph
        from them.

        Paramters
        ---------
        path_list : List[str]
            Paths to swc files on the local machine.

        Returns
        -------
        List[networkx.Graph]
            Attributed graphs.

        """
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Assign processes
            processes = list()
            for path in path_list:
                processes.append(
                    executor.submit(self.load_from_local_path, path)
                )

            # Store results
            graphs = list()
            for process in as_completed(processes):
                result = process.result()
                if result:
                    graphs.append(result)
        return graphs

    def load_from_local_path(self, path):
        """
        Reads a single swc file on local machine and builds an attributed
        graph from it.

        Paramters
        ---------
        path : str
            Path to swc file on the local machine.

        Returns
        -------
        networkx.Graph
            Attributed graph.

        """
        content = util.read_txt(path)
        if len(content) > self.min_size - 10:
            graph = self.parse(content)
            graph.graph["swc_id"] = util.get_swc_id(path)
            return graph
        else:
            return None

    def load_from_local_zip(self, zip_path):
        """
        Reads swc files from a zip file and builds attributed graphs from
        them.

        Paramters
        ---------
        zip_path : str
            Path to zip file to be read.

        Returns
        -------
        List[networkx.Graph]
            Attributed graphs.

        """
        with ZipFile(zip_path, "r") as zip_file:
            graphs = list()
            swc_files = [f for f in zip_file.namelist() if f.endswith(".swc")]
            for f in tqdm(swc_files, desc="Loading Fragments"):
                result = self.load_from_zip(zip_file, f)
                if result:
                    graphs.append(result)
        return graphs

    def load_from_gcs(self, gcs_dict):
        """
        Reads swc files from zips on a GCS bucket and builds attributed
        graphs from them.

        Parameters
        ----------
        gcs_dict : dict
            Dictionary with the keys "bucket_name" and "path" used to read
            swcs from GCS bucket

        Returns
        -------
        List[networkx.Graph]
            Attributed graphs.

        """
        bucket = storage.Client().bucket(gcs_dict["bucket_name"])
        zip_paths = util.list_gcs_filenames(bucket, gcs_dict["path"], ".zip")
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for path in tqdm(zip_paths, desc="Download SWCs"):
                zip_bytes = bucket.blob(path).download_as_bytes()
                processes.append(
                    executor.submit(self.load_from_cloud_zip, zip_bytes)
                )

            # Store results
            graphs = list()
            for process in as_completed(processes):
                graphs.extend(process.result())
        return graphs

    def load_from_cloud_zip(self, zip_bytes):
        """
        Reads swc files from a zip and builds attributed graphs from them.

        Parameters
        ----------
        zip_bytes : bytes
            Contents of a zip file in byte format.

        Returns
        -------
        List[networkx.Graph]
            Attributed graphs.

        """
        with ZipFile(BytesIO(zip_bytes)) as zip_file:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = list()
                for f in util.list_files_in_zip(zip_bytes):
                    threads.append(
                        executor.submit(
                            self.load_from_zip, zip_file, f
                        )
                    )

                # Process results
                graphs = list()
                for thread in as_completed(threads):
                    result = thread.result()
                    if result:
                        graphs.append(result)
        return graphs

    def load_from_zip(self, zip_file, path):
        """
        Reads swc files at in a zip file at "path" and builds attributed
        graphs from them.

        Parameters
        ----------
        zip_file : ZipFile
            Zip containing swc file to be read.
        path : str
            Path to swc file to be read.

        Returns
        -------
        networkx.Graph
            Attributed graph.

        """
        content = util.read_zip(zip_file, path).splitlines()
        if len(content) > self.min_size - 10:
            graph = self.parse(content)
            graph.graph["swc_id"] = util.get_swc_id(path)
            return graph
        else:
            return False

    # --- Process SWC Contents ---
    def parse(self, content):
        """
        Reads an swc file and builds an attributed graphs from it.

        Parameters
        ----------
        path : str
            Path to swc file to be read.

        Returns
        -------
        networkx.Graph
            Graph built from an swc file.

        """
        graph = nx.Graph()
        content, offset = self.process_content(content)
        for line in content:
            # Extract node info
            parts = line.split()
            child = int(parts[0])
            parent = int(parts[-1])
            radius = self.read_radius(parts[-2])
            xyz = self.read_xyz(parts[2:5], offset=offset)

            # Add node
            graph.add_node(child, radius=radius, xyz=xyz)
            if parent != -1:
                graph.add_edge(parent, child)
        return graph

    def process_content(self, content):
        """
        Processes lines of text from an swc file by iterating over commented
        lines to extract offset (if present) and finds the line after the last
        commented line.

        Parameters
        ----------
        content : List[str]
            List of strings that represent a line of a text file.

        Returns
        -------
        List[str]
            A list of strings representing the lines of text starting from the
            line immediately after the last commented line.
        List[float]
            Offset of swc file.

        """
        offset = [0.0, 0.0, 0.0]
        for i, line in enumerate(content):
            if line.startswith("# OFFSET"):
                offset = self.read_xyz(line.split()[2:5])
            if not line.startswith("#"):
                return content[i:], offset

    def read_xyz(self, xyz_str, offset=[0.0, 0.0, 0.0]):
        """
        Reads the coordinates from a string and transforms it (if applicable).

        Parameters
        ----------
        xyz_str : str
            Coordinate stored in a str.
        offset : list[int], optional
            Offset of coordinates in swc file. The default is [0.0, 0.0, 0.0].

        Returns
        -------
        numpy.ndarray
            xyz coordinates of an entry from an swc file.

        """
        xyz = np.zeros((3))
        for i in range(3):
            xyz[i] = self.anisotropy[i] * (float(xyz_str[i]) + offset[i])
        return xyz

    def read_radius(self, radius_str):
        """
        Converts a radius string to a float and adjusts it if the value is in
        nanometers.

        Parameters
        ----------
        radius_str : str
            A string representing the radius value.

        Returns
        -------
        float
            Radius.

        """
        radius = float(radius_str)
        return radius / 1000 if radius > 100 else radius


# --- Write ---
def write(path, content, color=None):
    """
    Writes an swc from the given "content" which is either a list of entries
    or a graph.

    Parameters
    ----------
    path : str
        Path where the content is to be written.
    content : List[str] or networkx.Graph
        Content of swc file to be written.
    color : str, optional
        Color of swc to be written. The default is None.

    Returns
    -------
    None

    """
    if type(content) is list:
        write_list(path, content, color=color)
    elif type(content) is nx.Graph:
        write_graph(path, content, color=color)
    else:
        raise Exception("Unable to write {} to swc".format(type(content)))


def write_list(path, entry_list, color=None):
    """
    Writes a list of swc entries to a file at path.

    Parameters
    ----------
    path : str
        Path that swc will be written to.
    entry_list : List[str]
        List of entries to be written to an swc file.
    color : str, optional
        Color of swc to be written. The default is None.

    Returns
    -------
    None.

    """
    with open(path, "w") as f:
        # Comments
        if color is not None:
            f.write("# COLOR " + color)
        else:
            f.write("# id, type, x, y, z, r, pid")

        # Entries
        for i, entry in enumerate(entry_list):
            f.write("\n" + entry)


def write_graph(path, graph, color=None):
    """
    Writes a graph to an swc file. This routine assumes that "graph" has a
    single connected component.

    Parameters
    ----------
    path : str
        Path that swc will be written to.
    graph : networkx.Graph
        Graph to be written to swc file.

    Returns
    -------
    None

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
