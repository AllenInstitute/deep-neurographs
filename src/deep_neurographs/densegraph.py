"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class of graphs built from swc files where each entry in the swc file
corresponds to a node in the graph.

"""

import os
from random import sample
from scipy.spatial import KDTree
from time import time

import networkx as nx

from deep_neurographs import swc_utils, utils

DELETION_RADIUS = 10

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
        self.init_graphs(swc_paths)
        self.init_kdtree()

    def init_graphs(self, paths):
        """
        Initializes graphs by reading swc files in "paths". Graphs are
        stored in a hash table where the entries are filename-graph pairs.

        Parameters
        ----------
        paths : list[str]
            List of paths to swc files that are used to construct a dictionary
            in which the items are filename-graph pairs.

        Returns
        -------
        None

        """
        self.graphs = dict()
        self.xyz_to_swc = dict()
        swc_dicts, _ = swc_utils.process_local_paths(paths)
        for i, swc_dict in enumerate(swc_dicts):
            swc_id = swc_dict["swc_id"]
            graph, _ = swc_utils.to_graph(swc_dict, set_attrs=True)
            self.store_xyz(graph, swc_id)
            self.graphs[swc_id] = graph

    def store_xyz(self, graph, swc_id):
        for i in graph.nodes:
            self.xyz_to_swc[tuple(graph.nodes[i]["xyz"])] = swc_id

    def init_kdtree(self):
        """
        Builds a KD-Tree from the xyz coordinates from every node stored in
        self.graphs.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.kdtree = KDTree(list(self.xyz_to_swc.keys()))

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])

    def save(self, output_dir):
        for swc_id, graph in self.graphs.items():
            cnt = 0
            for component in nx.connected_components(graph):
                entry_list = self.make_entries(graph, component)
                path = os.path.join(output_dir, f"{swc_id}.swc")
                while os.path.exists(path):
                    path = os.path.join(output_dir, f"{swc_id}.{cnt}.swc")
                    cnt += 1
                swc_utils.write(path, entry_list)

    def make_entries(self, graph, component):
        node_to_idx = dict()
        entry_list = []
        for i, j in nx.dfs_edges(graph.subgraph(component)):
            # Initialize
            if len(entry_list) == 0:
                node_to_idx[i] = 1
                x, y, z = tuple(graph.nodes[i]["xyz"])
                r = graph.nodes[i]["radius"]
                entry_list.append([1, 2, x, y, z, r, -1])

            # Create entry
            node_to_idx[j] = len(entry_list) + 1
            x, y, z = tuple(graph.nodes[j]["xyz"])
            r = graph.nodes[j]["radius"]
            entry_list.append([node_to_idx[j], 2, x, y, z, r, node_to_idx[i]])
        return entry_list
        
