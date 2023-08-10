"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "NeuroGraph".

"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from more_itertools import zip_broadcast
from scipy.spatial import KDTree
from deep_neurographs import graph_utils as gutils, swc_utils, utils

COLORS = list(mcolors.TABLEAU_COLORS.keys())
nCOLORS = len(COLORS)


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files. This type of graph has two sets of edges referred
    to as "mutable" and "immutable".

    """

    def __init__(self, max_mutable_degree=5, max_mutable_edge_dist=150.0, prune=True, prune_depth=10):
        """
        Parameters
        ----------
        max_mutable_degree : int, optional
            Maximum degree of any given node with respect to the number of
            mutable edges. The default is 5.
        max_mutable_edge_dist : float, optional
            Maximum between any two nodes that form a mutable edge.
            The default is 200.

        Returns
        -------
        None.

        """
        super(NeuroGraph, self).__init__()
        # Parameters
        self.prune = prune
        self.prune_depth = prune_depth
        self.max_mutable_degree = max_mutable_degree
        self.max_mutable_edge_dist = max_mutable_edge_dist

        # Initializations
        self.leafs = set()
        self.junctions = set()
        self.mutable_edges = set()
        self.immutable_edges = set()
        self.xyz_to_edge = dict()

    # --- Add nodes or edges ---
    def generate_immutables(self, swc_id, swc_dict):
        """
        Adds nodes to graph from a dictionary generated from an swc files.

        Parameters
        ----------
        node_id : int
            Node id.
        swc_dict : dict
            Dictionary generated from an swc where the keys are swc type
            attributes.

        Returns
        -------
        None.

        """
        # Add nodes
        leafs, junctions, edges = gutils.extract_irreducible_graph(swc_dict)
        node_id = dict()
        for i in leafs + junctions:
            node_id[i] = len(self.nodes)
            self.add_node(
                node_id[i],
                xyz=swc_dict["xyz"][i],
                radius=swc_dict["radius"][i],
                swc_id=swc_id,
            )

        # Add edges
        for (i, j) in edges.keys():
            edge = (node_id[i], node_id[j])
            self.immutable_edges.add(frozenset(edge))
            self.add_edge(
                node_id[i],
                node_id[j],
                xyz=np.array(edges[(i,j)]["xyz"]),
                radius=np.array(edges[(i,j)]["radius"]),
                swc_id=swc_id,
            )
            xyz_to_edge = dict([(xyz, edge) for xyz in edges[(i, j)]["xyz"]])
            self.xyz_to_edge.update(xyz_to_edge)

        # Update leafs and junctions
        for l in leafs:
            self.leafs.add(node_id[l])

        for j in junctions:
            self.junctions.add(node_id[j])

    def generate_mutables(self):
        """
        Generates edges for the graph.

        Returns
        -------
        None

        """
        # Search for mutable connections
        self._init_kdtree()
        for leaf in self.leafs:
            xyz_leaf = self.nodes[leaf]["xyz"]
            for xyz in self._get_mutables(leaf, xyz_leaf):
                # Extract info on mutable connection                
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Get connecting node
                if utils.dist(xyz, attrs["xyz"][0]) < 32:
                    node = i
                elif utils.dist(xyz, attrs["xyz"][-1]) < 32:
                    node = j
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.add_immutable_node((i, j), attrs, idxs[0])

                # Add edge
                self.add_edge(leaf, node, xyz=np.array([xyz_leaf, xyz]))
                self.mutable_edges.add(frozenset((leaf, node)))

    def _get_mutables(self, query_id, query_xyz):
        """
        Parameters
        ----------
        query_id : int
            Node id of the query node.
        query_xyz : tuple[float]
            The (x,y,z) coordinates of the query node.

        Returns
        -------
        None.

        """
        # Search for connections
        query_edge = self.xyz_to_edge[tuple(query_xyz)]
        query_swc_id = gutils.get_edge_attr(self, query_edge, "swc_id")
        best_xyz = dict()
        best_dist = dict()
        for xyz in self._query_kdtree(query_xyz):
            xyz = tuple(xyz.astype(int))
            edge = self.xyz_to_edge[xyz]
            swc_id = gutils.get_edge_attr(self, edge, "swc_id")
            if swc_id != query_swc_id:
                d = utils.dist(xyz, query_xyz)
                if edge not in best_dist.keys():
                    best_xyz[edge] = xyz
                    best_dist[edge] = d
                elif d < best_dist[edge]:
                    best_xyz[edge] = xyz
                    best_dist[edge] = d
        return self._get_best_edges(best_dist, best_xyz)

    def _get_best_edges(self, dist, xyz):
        """
        Gets the at most "self.max_degree" nodes that are closest to the
        target node.

        Parameters
        ----------
        best_dist : dict
            Dictionary where the keys are node_ids and values are distance
            from the target node.

        Returns
        -------
        best_dist : dict
            Dictionary of nodes that are closest to the target node.

        """
        if len(dist.keys()) > self.max_mutable_degree:
            keys = sorted(dist, key=dist.__getitem__)
            return [xyz[key] for key in keys[:self.max_mutable_degree]]
        else:
            return list(xyz.values())

    def add_immutable_node(self, edge, attrs, idx):
        # Remove old edge
        (i, j) = edge
        self.remove_edge(i, j)
        self.immutable_edges.remove(frozenset(edge))

        # Add new node and split edge
        node_id = len(self.nodes)
        self.add_node(
            node_id,
            xyz=tuple(attrs["xyz"][idx]),
            radius=attrs["radius"][idx],
            swc_id=attrs["swc_id"],
        )
        self._add_edge((i, node_id), attrs, np.arange(0, idx+1))
        self._add_edge((node_id, j), attrs, np.arange(idx, len(attrs["xyz"])))
        return node_id
    
    def _add_edge(self, edge, attrs, idxs):
        self.add_edge(
            edge[0],
            edge[1],
            xyz=attrs["xyz"][idxs],
            radius=attrs["radius"][idxs],
            swc_id=attrs["swc_id"],
        )
        for xyz in attrs["xyz"][idxs]:
            self.xyz_to_edge[tuple(xyz)] = edge
        self.immutable_edges.add(frozenset(edge))

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
        self.kdtree = KDTree(list(self.xyz_to_edge.keys()))

    def _query_kdtree(self, query):
        """
        Parameters
        ----------
        query : int
            Node id.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "self.max_edge_dist" from "query".

        """
        idxs = self.kdtree.query_ball_point(query, self.max_mutable_edge_dist)
        return self.kdtree.data[idxs]

    # --- Visualization ---
    def _init_figure(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig, ax

    def visualize_immutables(self):
        """
        Parameters
        ----------
        node_ids : list[int], optional
            List of node ids to be plotted. The default is None.
        edge_ids : list[tuple], optional
            List of edge ids to be plotted. The default is None.

        Returns
        -------
        None.

        """
        _, ax = self._init_figure()
        self._plot_edges(ax, self.immutable_edges)
        plt.show()

    def visualize_mutables(self):
        _, ax = self._init_figure()
        self._plot_edges(ax, self.immutable_edges, color="k")
        self._plot_edges(ax, self.mutable_edges)
        plt.show()

    def _plot_node(self, ax, i, color="r"):
        ax.scatter(
            self.nodes[i]["xyz"][0],
            self.nodes[i]["xyz"][1],
            self.nodes[i]["xyz"][2],
            color=color,
        )

    def _plot_edges(self, ax, edges, color=None):
        for (i, j) in edges:
            ax.plot(
                self.edges[(i, j)]["xyz"][:, 0],
                self.edges[(i, j)]["xyz"][:, 1],
                self.edges[(i, j)]["xyz"][:, 2],
                color=color,
            )
            self._plot_node(ax, i)
            self._plot_node(ax, j)
        
        
    # --- Utils ---
    def num_nodes(self):
        """
        Computes number of nodes in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of nodes in the graph.

        """
        return self.number_of_nodes()

    def num_edges(self):
        """
        Computes number of edges in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of edges in the graph.

        """
        return self.number_of_edges()
    
    def to_line_graph(self):
        """
        Converts graph to a line graph.

        Parameters
        ----------
        None

        Returns
        -------
        networkx.Graph
            Line graph.

        """
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.edges)
        return nx.line_graph(graph)
