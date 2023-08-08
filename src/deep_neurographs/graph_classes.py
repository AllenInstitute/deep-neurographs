"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "SuperGraph".

"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from more_itertools import zip_broadcast
from scipy.spatial import KDTree

from deep_neurographs import swc_utils, utils

COLORS = list(mcolors.TABLEAU_COLORS.keys())
nCOLORS = len(COLORS)


class SuperGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to super voxels in a predicted
    segmentation.

    """

    def __init__(self, max_degree=5, max_edge_dist=200.0):
        """
        Parameters
        ----------
        max_degree : int, optional
            Maximum degree of any given node. The default is 5.
        max_edge_dist : float, optional
            Maximum between any two nodes that form an edge.
            The default is 200.

        Returns
        -------
        None.

        """
        super(SuperGraph, self).__init__()
        self.max_degree = max_degree
        self.max_edge_dist = max_edge_dist
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
            Node id.
        swc_dict : dict
            Dictionary generated from an swc where the keys are swc type
            attributes.

        Returns
        -------
        None.

        """
        swc_keys = set(["parents", "radius", "subnodes", "xyz"])
        assert swc_keys == set(
            swc_dict.keys()
        ), "swc_dict must have the keys {}".format(swc_keys)
        leafs, junctions = swc_utils.extract_topo_nodes(
            swc_dict["subnodes"], swc_dict["parents"]
        )
        self.add_node(
            node_id,
            junctions=junctions,
            leafs=leafs,
            parents=swc_dict["parents"],
            radius=swc_dict["radius"],
            subnodes=swc_dict["subnodes"],
            xyz=swc_dict["xyz"],
        )

    def create_edges(self):
        """
        Generates edges for the graph.

        Returns
        -------
        None

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
        query : int
            Node id.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "self.max_edge_dist" from "query".

        """
        idxs = self.kdtree.query_ball_point(query, self.max_edge_dist)
        xyz = self.kdtree.data[idxs].tolist()
        return map(tuple, xyz)

    def _extract_edges(self, query_id, query_xyz, nbs_xyz):
        """
        Parameters
        ----------
        query_id : int
            Node id of the query node.
        query_xyz : tuple[float]
            The (x,y,z) coordinates of the query node.
        nbs_xyz : generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "self.max_edge_dist" from "query".

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
            self._add_edge(
                query_id,
                query_xyz,
                nb_id,
                best_xyz[nb_id],
                best_dist[nb_id],
            )

    def _get_best_edges(self, best_dist):
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
        if len(best_dist.keys()) > self.max_degree:
            sorted_keys = sorted(best_dist, key=best_dist.__getitem__)
            for key in sorted_keys[self.max_degree :]:
                del best_dist[key]
        return best_dist

    def _add_edge(self, id1, xyz1, id2, xyz2, dist):
        """
        Adds the edge {id1, id2} to the graph.

        Parameters
        ----------
        id1 : int
            Node id.
        xyz1 : tuple(float)
            The (x,y,z) coordinates of node "id1".
        id2 : int
            Node id.
        xyz2 : tuple(float)
            The (x,y,z) coordinates of node "id1".
        dist : float
            Distance between nodes "id1" and "id2".

        Returns
        -------
        None.

        """
        if self._check_to_add_edge(id1, id2, dist):
            self.add_edge(id1, id2, distance=dist, xyz={id1: xyz1, id2: xyz2})

    def _check_to_add_edge(self, id1, id2, dist):
        """
        Checks whether the edge {id1, id2} already exists. If so, the edge
        with a smaller distance is kept.

        Parameters
        ----------
        id1 : int
            Node id.
        id2 : int
            Node id.
        dist : float
            Distance between nodes "id1" and "id2".

        Returns
        -------
        bool
            Indication of whether to add proposed edge.

        """
        edge = (id1, id2)
        if self.has_edge(*edge):
            if self.edges[edge]["distance"] < dist:
                return False
        return True

    # --- Graph/Node/Edge Attribute routines ---
    def upd_xyz_to_id(self, node_id):
        """
        Updates self attribute "xyz_to_id" to include the newly added node
        (i.e. "node_id").

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        None.

        """
        self.node_exists(node_id)
        new_entries = dict(zip_broadcast(self.nodes[node_id]["xyz"], node_id))
        self.xyz_to_id.update(new_entries)

    # --- Visualization ---
    def _init_figure(self):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig, ax

    def visualize_node(self, node_id):
        """
        Generates a plot of node "node_id" that includes of the subnodes.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        None.

        """
        _, ax = self._init_figure()
        for (x, y, z), label in self._get_subnodes_data(node_id):
            ax.scatter(x, y, z)
            ax.text(x, y, z, label)
        plt.show()

    def _get_subnodes_data(self, node_id):
        self.node_exists(node_id)
        xyz = self.nodes[node_id]["xyz"]
        labels = self.nodes[node_id]["subnodes"]
        return list(zip(xyz, labels))

    def visualize_supergraph(self, node_ids=None, edge_ids=None):
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
        nodes = self.nodes if node_ids is None else node_ids
        for i, node_id in enumerate(nodes):
            color = COLORS[i % nCOLORS]
            for (x, y, z), _ in self._get_subnodes_data(node_id):
                ax.scatter(x, y, z, c=color)
        plt.show()

    # --- I/O Routines ---
    def write_swc(self):
        """
        Writes graph to swc file

        Returns
        -------
        None.

        """
        pass

    # --- Utils ---
    def node_exists(self, node_id):
        """
        Determines with node "node_id" exists in graph.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        None.

        """
        assert node_id in self.nodes, "Node does not exist!"

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
