"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclasses of Networkx.Graph called "SuperNode"
and "SuperGraph".

"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from more_itertools import zip_broadcast
from mpl_toolkits.mplot3d import Axes3D


class SuperGraph(nx.Graph):
    """
    A graph whose nodes are a set of "SuperNodes". 
    Note: Assume 
    """
    def __init__(self):
        super(SuperGraph, self).__init__()
        # Feature vectors
        self.node_features = []
        self.edge_features = []

        # Attributes
        self.node_attributes = ["subnodes", "xyz", "radius", "leafs", "junctions"]
        self.edge_attributes = []
        self.init_node_attributes()
        self.init_edge_attributes()

        # Book keeping
        self.old_node_ids = dict()
        self.xyz_to_id = dict()

    def init_node_attributes(self):
        for attr in self.node_attributes:
            nx.set_node_attributes(self, dict(), attr)

    def init_edge_attributes(self):
        for attr in self.node_attributes:
            nx.set_edge_attributes(self, dict(), attr)

    def set_node_attribute(self, node_id, key, value):
        assert key in self.node_attributes, "Node attribute does not exist!"
        self.nodes[node_id][key] = value

    def set_edge_attribute(self):
        pass

    def add_node_feature(self, feature):
        self.node_features.append(feature)
        assert len(self.node_features) == len(self.nodes), "Number of nodes and node features not equal!"

    def upd_xyz_to_id(self, node_id):
        assert node_id in self.nodes, "Node does not exist!"
        new_entries = dict(zip_broadcast(self.nodes[node_id]["xyz"], node_id))
        self.xyz_to_id.update(new_entries)

    def visualize_supernode(self, node_id):
        # Extract data
        assert node_id in self.nodes, "Node does not exist!"
        xyz = self.nodes[node_id]["xyz"]
        labels = self.nodes[node_id]["subnodes"] #list(map(str, )))
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

    def visualize_supergraph(self):
        pass

    def write_swc(self):
        pass
