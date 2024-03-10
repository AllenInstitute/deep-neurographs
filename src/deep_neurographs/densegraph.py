"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class of graphs built from swc files. Each swc file is stored as a distinct
graph and each node in this graph.

"""

import networkx as nx
import numpy as np
from more_itertools import zip_broadcast
from scipy.spatial import KDTree

from deep_neurographs import swc_utils, utils
from deep_neurographs.geometry import dist as get_dist


class DenseGraph:
    """
    Class of graphs built from swc files. Each swc file is stored as a
    distinct graph and each node in this graph.

    """

    def __init__(self, swc_paths):
        """
        Constructs a DenseGraph object from a directory of swc files.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files which are used to construct a hash
            table in which the entries are filename-graph pairs.

        Returns
        -------
        None

        """
        self.xyz_to_node = dict()
        self.xyz_to_swc = dict()
        self.init_graphs(swc_paths)
        self.init_kdtree()

    def init_graphs(self, swc_paths):
        """
        Initializes graphs by reading swc files in "swc_paths". Graphs are
        stored in a hash table where the entries are filename-graph pairs.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files which are used to construct a hash
            table in which the entries are filename-graph pairs.

        Returns
        -------
        None

        """
        self.graphs = dict()
        for path in swc_paths:
            # Construct Graph
            swc_dict = swc_utils.parse_local_swc(path)
            graph, xyz_to_node = swc_utils.to_graph(swc_dict, set_attrs=True)

            # Store
            swc_id = swc_dict["swc_id"]
            print(swc_id, path)
            if type(swc_dict["xyz"]) == np.ndarray:
                swc_dict["xyz"] = utils.numpy_to_hashable(swc_dict["xyz"])
            xyz_to_id = dict(zip_broadcast(swc_dict["xyz"], swc_id))
            self.graphs[swc_id] = graph
            self.xyz_to_node[swc_id] = xyz_to_node
            self.xyz_to_swc.update(xyz_to_id)

    def init_kdtree(self):
        """
        Initializes KDTree from all xyz coordinates contained in all
        graphs in "self.graphs".

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.kdtree = KDTree(list(self.xyz_to_swc.keys()))

    def query_kdtree(self, xyz):
        """
        Queries "self.kdtree" for the nearest neighbor of "xyz".

        Parameters
        ----------
        xyz : tuple[float]
            Coordinate to be queried.

        Returns
        -------
        tuple[float]
            Result of query.

        """
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])
