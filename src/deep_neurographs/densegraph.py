"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class of graphs built from swc files where each entry in the swc file
corresponds to a node in the graph.

"""

import os

import networkx as nx
from scipy.spatial import KDTree

from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import img_util, swc_util

DELETION_RADIUS = 10


class DenseGraph:
    """
    Class of graphs built from swc files. Each swc file is stored as a
    distinct graph and each node in this graph.

    """

    def __init__(self, swc_paths, img_patch_origin=None, img_patch_shape=None):
        """
        Constructs a DenseGraph object from a directory of swc files.

        Parameters
        ----------
        swc_paths : list[str]
            List of paths to swc files which are used to construct a hash
            table in which the entries are filename-graph pairs.
        ...

        Returns
        -------
        None

        """
        self.bbox = img_util.get_bbox(img_patch_origin, img_patch_shape)
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
        swc_dicts = swc_util.Reader().load(paths)
        for i, swc_dict in enumerate(swc_dicts):
            # Build graph
            swc_id = swc_dict["swc_id"]
            graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)
            if self.bbox:
                graph = gutil.trim_branches(graph, self.bbox)

            # Store graph
            self.store_xyz_swc(graph, swc_id)
            self.graphs[swc_id] = graph

    def store_xyz_swc(self, graph, swc_id):
        """
        Stores (xyz, swc_id) as an item in the dictionary "self.xyz_to_swc".

        Parameters
        ----------
        graph : netowrkx.Graph
            Graph to parsed.
        swc_id : str
            swc_id corresponding to "graph".

        Returns
        -------
        None

        """
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
        """
        Projects "xyz" onto "self by finding the closest point.

        Parameters
        ----------
        xyz : numpy.ndarray
            xyz coordinate to be queried.

        Returns
        -------
        numpy.ndarray
            Projection of "xyz".

        """
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])

    def save(self, output_dir):
        """
        Saves "self" to an swc file.

        Parameters
        ----------
        output_dir : str
            Path to directory that swc files are written to.

        Returns
        -------
        None

        """
        for swc_id, graph in self.graphs.items():
            cnt = 0
            for component in nx.connected_components(graph):
                entry_list = self.make_entries(graph, component)
                path = os.path.join(output_dir, f"{swc_id}.swc")
                while os.path.exists(path):
                    path = os.path.join(output_dir, f"{swc_id}.{cnt}.swc")
                    cnt += 1
                swc_util.write(path, entry_list)

    def make_entries(self, graph, component):
        """
        Makes swc entries corresponding to nodes in "component".

        Parameters
        ----------
        graph : networkx.Graph
            Graph that "component" is a connected component of.
        component : set
            Connected component of "graph".

        Returns
        -------
        entry_list
            List of swc entries generated from nodes in "component".

        """
        node_to_idx = dict()
        entry_list = []
        for i, j in nx.dfs_edges(graph.subgraph(component)):
            # Initialize
            if len(entry_list) == 0:
                node_to_idx[i] = 1
                x, y, z = tuple(graph.nodes[i]["xyz"])
                r = graph.nodes[i]["radius"]
                entry_list.append(f"1 2 {x} {y} {z} {r} -1")

            # Create entry
            node_to_idx[j] = len(entry_list) + 1
            x, y, z = tuple(graph.nodes[j]["xyz"])
            r = graph.nodes[j]["radius"]
            entry_list.append(
                f"{node_to_idx[j]} 2 {x} {y} {z} {r} {node_to_idx[i]}"
            )
        return entry_list
