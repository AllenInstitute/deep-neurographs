"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with swc files.

"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from google.cloud import storage
from io import BytesIO
from tqdm import tqdm
from zipfile import ZipFile

import networkx as nx
import numpy as np
import os

from deep_neurographs.utils import util


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
        anisotropy : list[float], optional
            Image to world scaling factors applied to xyz coordinates to
            account for anisotropy of the microscope. The default is
            [1.0, 1.0, 1.0].
        min_size : int, optional
            Threshold on the number of nodes in swc file. Only swc files with
            more than "min_size" nodes are stored in "xyz_coords". The default
            is 0.

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.min_size = min_size

    def load(self, swc_pointer):
        """
        Load data based on the type and format of the provided "swc_pointer".

        Parameters
        ----------
        swc_pointer : dict, list, str
            Object that points to swcs to be read, see class documentation for
            details.

        Returns
        -------
        list[dict]
            List of dictionaries whose keys and values are the attribute name
            and values from an swc file.

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
                paths = util.list_paths(swc_pointer, ext=".swc")
                return self.load_from_local_paths(paths)
        raise Exception("SWC Pointer is not Valid!")

    # --- Load subroutines ---
    def load_from_local_paths(self, swc_paths):
        """
        Reads swc files from local machine, then returns either the xyz
        coordinates or graphs.

        Paramters
        ---------
        swc_paths : list
            List of paths to swc files stored on the local machine.

        Returns
        -------
        list[dict]
            List of dictionaries whose keys and values are the attribute name
            and values from an swc file.

        """
        with ProcessPoolExecutor(max_workers=1) as executor:
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
        Reads a single swc file from local machine, then returns either the
        xyz coordinates or graphs.

        Paramters
        ---------
        path : str
            Path to swc file stored on the local machine.

        Returns
        -------
        list[dict]
            List of dictionaries whose keys and values are the attribute name
            and values from an swc file.

        """
        content = util.read_txt(path)
        if len(content) > self.min_size:
            result = self.parse(content)
            result["swc_id"] = util.get_swc_id(path)
            return result
        else:
            return False

    def load_from_gcs(self, gcs_dict):
        """
        Reads swc files from zips on a GCS bucket.

        Parameters
        ----------
        gcs_dict : dict
            Dictionary where keys are "bucket_name" and "path".

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates read from
            that swc file.

        """
        # Initializations
        bucket = storage.Client().bucket(gcs_dict["bucket_name"])
        zip_paths = util.list_gcs_filenames(bucket, gcs_dict["path"], ".zip")

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = []
            for path in zip_paths:
                zip_content = bucket.blob(path).download_as_bytes()
                processes.append(
                    executor.submit(self.load_from_cloud_zip, zip_content)
                )

            # Store results
            swc_dicts = list()
            desc = "Downloading SWCs"
            for process in tqdm(as_completed(processes), desc=desc):
                swc_dicts.extend(process.result())
        return swc_dicts

    def load_from_cloud_zip(self, zip_content):
        """
        Reads swc files from a zip that has been downloaded from a cloud
        bucket.

        Parameters
        ----------
        zip_content : ...
            content of a zip file.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates read from
            that swc file.

        """
        with ZipFile(BytesIO(zip_content)) as zip_file:
            with ThreadPoolExecutor() as executor:
                # Assign threads
                threads = []
                for f in util.list_files_in_zip(zip_content):
                    threads.append(
                        executor.submit(
                            self.load_from_cloud_zipped_file, zip_file, f
                        )
                    )

                # Process results
                swc_dicts = list()
                for thread in as_completed(threads):
                    result = thread.result()
                    if result:
                        swc_dicts.append(result)
        return swc_dicts

    def load_from_cloud_zipped_file(self, zip_file, path):
        """
        Reads swc file stored at "path" which points to a file in a zip.

        Parameters
        ----------
        zip_file : ZipFile
            Zip containing swc file to be read.
        path : str
            Path to swc file to be read.

        Returns
        -------
        dict
            Dictionary that maps an swc_id to the the xyz coordinates or graph
            read from that swc file.

        """
        content = util.read_zip(zip_file, path).splitlines()
        if len(content) > self.min_size:
            result = self.parse(content)
            result["swc_id"] = util.get_swc_id(path)
            return result
        else:
            return False

    # --- Process swc content ---
    def parse(self, content):
        """
        Parses an swc file to extract the content which is stored in a dict. Note
        that node_ids from swc are refactored to index from 0 to n-1 where n is
        the number of entries in the swc file.

        Parameters
        ----------
        content : list[str]
            List of entries from an swc file.

        Returns
        -------
        dict
            Dictionaries whose keys and values are the attribute name
            and values from an swc file.

        """
        # Parse swc content
        content, offset = self.process_content(content)
        swc_dict = {
            "id": np.zeros((len(content)), dtype=np.int32),
            "radius": np.zeros((len(content)), dtype=np.float32),
            "pid": np.zeros((len(content)), dtype=np.int32),
            "xyz": np.zeros((len(content), 3), dtype=np.float32),
        }
        for i, line in enumerate(content):
            parts = line.split()
            swc_dict["id"][i] = parts[0]
            swc_dict["radius"][i] = float(parts[-2])
            swc_dict["pid"][i] = parts[-1]
            swc_dict["xyz"][i] = self.read_xyz(parts[2:5], offset)

        # Check whether radius is in nanometers
        if swc_dict["radius"][0] > 100:
            swc_dict["radius"] /= 1000
        return swc_dict

    def process_content(self, content):
        """
        Processes lines of text from a content source, extracting an offset
        value and returning the remaining content starting from the line
        immediately after the last commented line.

        Parameters
        ----------
        content : List[str]
            List of strings where each string represents a line of text.

        Returns
        -------
        List[str]
            A list of strings representing the lines of text starting from the
            line immediately after the last commented line.
        List[float]
            Offset of swc file.

        """
        offset = [1.0, 1.0, 1.0]
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


def write(path, content, color=None):
    if type(content) is list:
        write_list(path, content, color=color)
    elif type(content) is dict:
        write_dict(path, content, color=color)
    elif type(content) is nx.Graph:
        write_graph(path, content, color=color)
    else:
        assert True, "Unable to write {} to swc".format(type(content))


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
            xyz = util.numpy_to_hashable(swc_dict["xyz"])
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