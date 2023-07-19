"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "SuperGraph".

"""

import itertools
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deep_neurographs import swc_utils, utils
from more_itertools import zip_broadcast
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree


COLORS = list(mcolors.TABLEAU_COLORS.keys())
nCOLORS = len(COLORS)


class SuperGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to super voxels in a predicted
    segmentation. 

    """
    def __init__(self, max_degree=5, max_edge_dist=200):
        """
        Parameters
        ----------
        max_degree : TYPE, optional
            DESCRIPTION. The default is 5.
        max_edge_dist : TYPE, optional
            DESCRIPTION. The default is 200.

        Returns
        -------
        None.

        """
        super(SuperGraph, self).__init__()
        # Feature vectors
        self.node_features = []
        self.edge_features = []

        # Graph properties
        self.max_degree = max_degree
        self.max_edge_dist = max_edge_dist

        # Book keeping
        self.old_node_ids = dict()
        self.xyz_to_id = dict()

    # --- Add node or edge ---
    def add_node_from_swc(self, node_id, swc_dict):
        """
        Adds a node to the graph. The node's attributes are stored in
        "swc_dict" which is generated from the corresponding swc file. 

        Parameters
        ----------
        node_id : int
            DESCRIPTION.
        swc_dict : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        swc_keys = set(["parents", "radius", "subnodes", "xyz"])
        assert swc_keys == set(swc_dict.keys()), "swc_dict must have the keys {}".format(swc_keys)
        leafs, junctions = swc_utils.extract_topo_nodes(swc_dict["subnodes"], swc_dict["parents"])
        self.add_node(
            node_id,
            junctions=junctions,
            leafs=leafs,
            radius=swc_dict["radius"],
            subnodes=swc_dict["subnodes"],
            xyz=swc_dict["xyz"],
        )

    def create_edges(self):
        """
        Returns
        -------
        bool
            DESCRIPTION.

        """
        self._init_kdtree()
        for query_id in self.nodes:
            for leaf in self.nodes[query_id]["leafs"]:
                query_xyz = self.nodes[query_id]["xyz"][leaf]
                nbs_xyz = self._query_kdtree(query_xyz)
                self._extract_edges(query_id, query_xyz, nbs_xyz)

    def _init_kdtree(self):
        """
        Builds a KD-Tree from the (x,y,z) coordinates of the subnodes of
        each node in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.kdtree = KDTree(list(self.xyz_to_id.keys()))

    def _query_kdtree(self, query):
        """
        Parameters
        ----------
        query : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        idxs = self.kdtree.query_ball_point(query, self.max_edge_dist)
        xyz = self.kdtree.data[idxs].tolist()
        return map(tuple, xyz)

    def _extract_edges(self, query_id, query_xyz, nbs_xyz):
        """
        Parameters
        ----------
        query_id : TYPE
            DESCRIPTION.
        query_xyz : TYPE
            DESCRIPTION.
        nbs_xyz : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Search for best edges
        best_xyz = dict()
        best_dist = dict()
        while True:
            try:
                xyz = next(nbs_xyz)
            except StopIteration:
                break

            if self.xyz_to_id[xyz] != query_id:
                d = utils.dist(xyz, query_xyz)
                if not utils.check_key(best_xyz, self.xyz_to_id[xyz]):
                    best_xyz[self.xyz_to_id[xyz]] = xyz
                    best_dist[self.xyz_to_id[xyz]] = d
                elif d < best_dist[self.xyz_to_id[xyz]]:
                    best_xyz[self.xyz_to_id[xyz]] = xyz
                    best_dist[self.xyz_to_id[xyz]] = d

        # Add best edges
        best_dist = self._get_best_edges(best_dist)
        for nb_id in best_dist.keys():
            print("nb_id:", nb_id)
            print("Add edge from {} to {}".format(query_id, nb_id))
            print("xyz coordinates are {} and {}".format(query_xyz, best_xyz[nb_id]))
            print("dist =", best_dist[nb_id])
            self._add_edge(
                query_id,
                query_xyz,
                nb_id, 
                best_xyz[nb_id],
                best_dist[nb_id],
                )
            print("")

    def _get_best_edges(self, best_dist):
        """
        Gets the at most "self.max_degree" nodes that are closest to the
        target node.

        Parameters
        ----------
        best_dist : dicts
            Dictionary where the keys are node_ids and values are distance
            from the target node.

        Returns
        -------
        best_dist : dict
            Dictionary of nodes that are closest to the target node.

        """
        if len(best_dist.keys()) > self.max_degree:
            sorted_keys = sorted(best_dist, key=best_dist.__getitem__)
            for key in sorted_keys[self.max_degree::]:
                del best_dist[key]
        return best_dist

    def _add_edge(self, id1, xyz1, id2, xyz2, dist):
        """
        

        Parameters
        ----------
        id1 : int
            DESCRIPTION.
        xyz1 : tuple(float)
            DESCRIPTION.
        id2 : int
            DESCRIPTION.
        xyz2 : tuple(float)
            DESCRIPTION.
        dist : float
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self._check_to_add_edge(id1, id2, dist):
            print("--> Added edge ({}, {})".format(id1, id2))
            self.add_edge(id1, id2, distance=dist, xyz={id1: xyz1, id2: xyz2})        

    def _check_to_add_edge(self, id1, id2, dist):
        """
        

        Parameters
        ----------
        id1 : TYPE
            DESCRIPTION.
        id2 : TYPE
            DESCRIPTION.
        dist : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        edge = (id1, id2)
        if self.has_edge(*edge):
            if self.edges[edge]["distance"] < dist:
                return False
            print("Updating edge ({}, {}) because better match found".format(id1, id2))
        return True
        
    # --- Graph/Node/Edge Attribute routines ---
    def upd_xyz_to_id(self, node_id):
        """


        Parameters
        ----------
        node_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.check_node_id(node_id)
        new_entries = dict(zip_broadcast(self.nodes[node_id]["xyz"], node_id))
        self.xyz_to_id.update(new_entries)

    # --- Add features ---
    def add_node_feature(self, feature):
        """
        Parameters
        ----------
        feature : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.node_features.append(feature)
        assert len(self.node_features) == len(self.nodes), "Number of nodes and node features not equal!"

    def add_edge_feature(self, feature):
        """
        Parameters
        ----------
        feature : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    # --- Visualization ---
    def _init_figure(self):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig, ax

    def visualize_node(self, node_id):
        """
        Generates a plot of node "node_id" that includes of the subnodes.

        Parameters
        ----------
        node_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        fig, ax = self._init_figure()
        for (x, y, z), label in self._get_subnodes_data(node_id):
            ax.scatter(x, y, z)
            ax.text(x, y, z, label)
        plt.show()

    def _get_subnodes_data(self, node_id):
        self.check_node_id(node_id)
        xyz = self.nodes[node_id]["xyz"]
        labels = self.nodes[node_id]["subnodes"]
        return list(zip(xyz, labels))

    def visualize_supergraph(self, node_ids=None, edge_ids=None):
        """
        Parameters
        ----------
        node_ids : TYPE, optional
            DESCRIPTION. The default is None.
        edge_ids : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        fig, ax = self._init_figure()
        nodes = self.nodes if node_ids is None else node_ids
        for i, node_id in enumerate(nodes):
            color = COLORS[i % nCOLORS]
            for (x, y, z), label in self._get_subnodes_data(node_id):
                ax.scatter(x, y, z, c=color)
        plt.show()

    # --- I/O Routines ---
    def write_swc(self):
        """
        Returns
        -------
        None.

        """
        pass

    # --- Assertions ---
    def check_node_id(self, node_id):
        """
        Parameters
        ----------
        node_id : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert node_id in self.nodes, "Node does not exist!"
