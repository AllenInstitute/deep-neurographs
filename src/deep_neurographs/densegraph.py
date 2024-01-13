"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Class of graphs built from swc files. Each swc file is stored as a distinct
graph and each node in this graph.

"""

import networkx as nx
from more_itertools import zip_broadcast
from scipy.spatial import KDTree

from deep_neurographs import swc_utils
from deep_neurographs.geometry_utils import dist as get_dist


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
            xyz_to_id = dict(zip_broadcast(swc_dict["xyz"], path))
            self.graphs[path] = graph
            self.xyz_to_node[path] = xyz_to_node
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

    def is_connected(self, xyz_1, xyz_2):
        """
        Determines whether the points "xyz_1" and "xyz_2" belong to the same
        swc file (i.e. graph).

        Parameters
        ----------
        xyz_1 : tuple[float]
            Coordinate contained in some graph in "self.graph".
        xyz_2 : tuple[float]
            Coordinate contained in some graph in "self.graph".

        Returns
        -------
        bool
            Indication of whether "xyz_1" and "xyz_2" belong to the same swc
            file (i.e. graph).

        """
        swc_identical = self.xyz_to_swc[xyz_1] == self.xyz_to_swc[xyz_2]
        return True if swc_identical else False

    def connect_nodes(self, xyz_1, xyz_2):
        """
        Finds path connecting two points that belong to some graph in
        "self.graph".

        Parameters
        ----------
        xyz_1 : tuple[float]
            Source of path.
        xyz_2 : tuple[float]
            Target of path.

        Returns
        -------
        list[int]
            Path of nodes connecting source and target.
        float
            Length of path with respect to l2-metric.

        """
        graph_id = self.xyz_to_swc[xyz_1]
        i = self.xyz_to_node[graph_id][xyz_1]
        j = self.xyz_to_node[graph_id][xyz_2]
        path = nx.shortest_path(self.graphs[graph_id], source=i, target=j)
        return path, self.path_length(graph_id, path)

    def path_length(self, graph_id, path):
        """
        Computes length of path with respect to the l2-metric.

        Parameters
        ----------
        graph_id : str
            ID of graph that path belongs to.
        path : list[int]
            List of nodes that form a path.
        Returns
        -------
        float
            Length of path with respect to l2-metrics.

        """
        path_length = 0
        for i in range(1, len(path)):
            path_length += get_dist(
                self.graphs[graph_id].nodes[i]["xyz"],
                self.graphs[graph_id].nodes[i - 1]["xyz"],
            )
        return path_length

    def is_aligned(self, xyz_1, xyz_2, ratio_threshold=0.5, exclude=10.0):
        """
        Determines whether the edge proposal corresponding to "xyz_1" and
        "xyz_2" is aligned to the ground truth. This is determined by checking
        two conditions: (1) connectedness and (2) distance ratio. For (1), we
        project "xyz_1" and "xyz_2" onto "self.graph", then verify that they
        project to the same graph. For (2), we compute the ratio between the
        Euclidean distance "dist" from "xyz_1" to "xyz" and the path length
        between the corresponding projections. This ratio can be skewed if
        "dist" is small, so we skip this criteria if "dist" < "exclude".

        Parameters
        ----------
        xyz_1 : numpy.array
            Endpoint of edge proposal.
        xyz_2 : numpy.array
            Endpoint of edge proposal.
        ratio_threshold : float
            Lower bound on threshold used to compare similarity between "dist"
            and "path length".
        exclude : float
            Upper bound on threshold to ignore criteria 1.

        Returns
        -------
        bool
            Indication of whether edge proposal is aligned to ground truth.

        """
        ratio = 0
        hat_xyz_1 = self.query_kdtree(xyz_1)
        hat_xyz_2 = self.query_kdtree(xyz_2)
        if self.is_connected(hat_xyz_1, hat_xyz_2):
            dist = get_dist(hat_xyz_1, hat_xyz_2)
            _, path_length = self.connect_nodes(hat_xyz_1, hat_xyz_2)
            dist = 1 if dist < 1 else dist
            path_length = 1 if path_length < 1 else path_length
            ratio = min(dist, path_length) / max(dist, path_length)
            if dist <= exclude:
                return True
            elif ratio > ratio_threshold:
                return True
            return True
        return False
