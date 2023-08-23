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
import plotly.graph_objects as go
import plotly.tools as tls
from more_itertools import zip_broadcast
from plotly.subplots import make_subplots
from scipy.spatial import KDTree

from deep_neurographs import graph_utils as gutils
from deep_neurographs import swc_utils, utils

COLORS = list(mcolors.TABLEAU_COLORS.keys())
nCOLORS = len(COLORS)


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files. This type of graph has two sets of edges referred
    to as "mutable" and "immutable".

    """

    def __init__(self):
        """
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        super(NeuroGraph, self).__init__()
        self.leafs = set()
        self.junctions = set()
        self.mutable_edges = set()
        self.immutable_edges = set()
        self.xyz_to_edge = dict()

    # --- Add nodes or edges ---
    def generate_immutables(
        self, swc_id, swc_dict, prune=True, prune_depth=16
    ):
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
        leafs, junctions, edges = gutils.extract_irreducible_graph(
            swc_dict, prune=prune, prune_depth=prune_depth
        )
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
                xyz=np.array(edges[(i, j)]["xyz"]),
                radius=np.array(edges[(i, j)]["radius"]),
                swc_id=swc_id,
            )
            xyz_to_edge = dict(
                (tuple(xyz), edge) for xyz in edges[(i, j)]["xyz"]
            )
            check_xyz = set(xyz_to_edge.keys())
            collisions = check_xyz.intersection(set(self.xyz_to_edge.keys()))
            if len(collisions) > 0:
                for xyz in collisions:
                    del xyz_to_edge[xyz]
            self.xyz_to_edge.update(xyz_to_edge)

        # Update leafs and junctions
        for l in leafs:
            self.leafs.add(node_id[l])

        for j in junctions:
            self.junctions.add(node_id[j])

    def generate_mutables(self, max_degree=5, max_dist=100.0):
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
            mutables = self._get_mutables(leaf, xyz_leaf, max_degree, max_dist)
            for xyz in mutables:
                # Extract info on mutable connection
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Get connecting node
                if utils.dist(xyz, attrs["xyz"][0]) < 16:
                    node = i
                    xyz = self.nodes[node]["xyz"]
                elif utils.dist(xyz, attrs["xyz"][-1]) < 16:
                    node = j
                    xyz = self.nodes[node]["xyz"]
                    if node == leaf:
                        stop
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.add_immutable_node((i, j), attrs, idxs[0])

                # Add edge
                self.add_edge(leaf, node, xyz=np.array([xyz_leaf, xyz]))
                if frozenset((leaf, node)) == frozenset({309}):
                    print((leaf, node))
                    stop
                self.mutable_edges.add(frozenset((leaf, node)))

    def _get_mutables(self, query_id, query_xyz, max_degree, max_dist):
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
        best_xyz = dict()
        best_dist = dict()
        query_swc_id = self.nodes[query_id]["swc_id"]
        for xyz in self._query_kdtree(query_xyz, max_dist):
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
        return self._get_best_edges(best_dist, best_xyz, max_degree)

    def _get_best_edges(self, dist, xyz, max_degree):
        """
        Gets the at most "max_degree" nodes that are closest to the
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
        if len(dist.keys()) > max_degree:
            keys = sorted(dist, key=dist.__getitem__)
            return [xyz[key] for key in keys[0:max_degree]]
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
        self._add_edge((i, node_id), attrs, np.arange(0, idx + 1))
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

    def _query_kdtree(self, query, max_dist):
        """
        Parameters
        ----------
        query : int
            Node id.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "max_dist" from "query".

        """
        idxs = self.kdtree.query_ball_point(query, max_dist)
        return self.kdtree.data[idxs]

    # --- Visualization ---
    def visualize_immutables(self, return_data=False):
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
        data = self._plot_edges(self.immutable_edges)
        data.append(self._plot_nodes())
        if return_data:
            return data
        else:
            utils.plot(data, "Immutable Graph")

    def visualize_mutables(self):
        data = [self._plot_nodes()]
        data.extend(self._plot_edges(self.immutable_edges, color="black"))
        data.extend(self._plot_edges(self.mutable_edges))
        utils.plot(data, "Mutable Graph")

    def _plot_nodes(self):
        xyz = nx.get_node_attributes(self, "xyz")
        xyz = np.array(list(xyz.values()))
        points = go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name="Nodes",
            marker=dict(size=3, color="red"),
        )
        return points

    def _plot_edges(self, edges, color=None):
        traces = []
        line = dict(width=4) if color is None else dict(color=color, width=3)
        for (i, j) in edges:
            trace = go.Scatter3d(
                x=self.edges[(i, j)]["xyz"][:, 0],
                y=self.edges[(i, j)]["xyz"][:, 1],
                z=self.edges[(i, j)]["xyz"][:, 2],
                mode="lines",
                line=line,
            )
            traces.append(trace)
        return traces

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
