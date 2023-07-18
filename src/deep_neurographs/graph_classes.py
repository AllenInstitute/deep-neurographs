"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclasses of Networkx.Graph called "SuperNode"
and "SuperGraph".

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
        """
        super(SuperGraph, self).__init__()
        # Feature vectors
        self.node_features = []
        self.edge_features = []

        # Attributes
        self.node_attributes = ["subnodes", "xyz", "radius", "leafs", "junctions"]
        self.edge_attributes = []

        # Graph properties
        self.max_degree = max_degree
        self.max_edge_dist = max_edge_dist

        # Book keeping
        self.old_node_ids = dict()
        self.xyz_to_id = dict()

    # --- Add node or edge ---
    def add_node_from_swc(self, node_id, swc_dict):
        """
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
        """
        idxs = self.kdtree.query_ball_point(query, self.max_edge_dist)
        xyz = self.kdtree.data[idxs].tolist()
        return map(tuple, xyz)

    def _extract_edges(self, query_id, query_xyz, nbs_xyz):
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
                #print(d)
                #print(xyz)
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
            print("")
            self._add_edge(query_id, query_xyz, nb_id,  best_xyz[nb_id])
        self.visualize_supergraph(node_ids=list(best_dist.keys()) + [0])
        print("Done")
        stop

    def _get_best_edges(self, best_dist):
        if len(best_dist.keys()) > self.max_degree:
            # to do...
            return best_dist
        else:
            return best_dist

    def _add_edge(self, id1, xyz1, id2, xyz2):
        # check whether edge has already been added
        # add edge to self by using id1 and id2
        # add xyz1 and xyz2 as attributes
        # generate features
        pass

    # --- Update attribute ---
    def upd_xyz_to_id(self, node_id):
        """
        """
        self.check_node_id(node_id)
        new_entries = dict(zip_broadcast(self.nodes[node_id]["xyz"], node_id))
        self.xyz_to_id.update(new_entries)

    # --- Add features ---
    def add_node_feature(self, feature):
        """
        """
        self.node_features.append(feature)
        assert len(self.node_features) == len(self.nodes), "Number of nodes and node features not equal!"

    def add_edge_feature(self, feature):
        """
        """
        pass

    # --- Visualization ---
    def visualize_supernode(self, node_id):
        """
        """
        # Extract data
        self.check_node_id(node_id)
        xyz = self.nodes[node_id]["xyz"]
        labels = self.nodes[node_id]["subnodes"]
        points = list(zip(xyz, labels))

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each point with labels
        for (x, y, z), label in points:
            ax.scatter(x, y, z)
            ax.text(x, y, z, label)

        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Show the plot
        plt.show()

    def visualize_supergraph(self, node_ids=None):
        """
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        nodes = self.nodes if node_ids is None else node_ids
        print("Plotting...")
        for i, node_id in enumerate(nodes):
            # Extract data
            print(node_id)
            self.check_node_id(node_id)
            xyz = self.nodes[node_id]["xyz"]
            labels = self.nodes[node_id]["subnodes"]
            points = list(zip(xyz, labels))

            # Plot each point with labels
            color = COLORS[i % nCOLORS]
            for (x, y, z), label in points:
                ax.scatter(x, y, z, c=color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    # --- I/O Routines ---
    def write_swc(self):
        """
        """
        pass

    # --- Assertions ---
    def check_node_id(self, node_id):
        assert node_id in self.nodes, "Node does not exist!"
