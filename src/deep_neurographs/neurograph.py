"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "NeuroGraph".

"""

from copy import deepcopy

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree

from deep_neurographs import geometry
from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.densegraph import DenseGraph
from deep_neurographs.geometry import check_dists
from deep_neurographs.geometry import dist as get_dist

SUPPORTED_LABEL_MASK_TYPES = [dict, np.array, ts.TensorStore]


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files.

    """

    def __init__(
        self, img_bbox=None, swc_paths=None, img_path=None, label_mask=None
    ):
        super(NeuroGraph, self).__init__()
        # Initialize paths
        self.img_path = img_path
        self.label_mask = label_mask
        self.swc_paths = swc_paths

        # Initialize node and edge sets
        self.leafs = set()
        self.junctions = set()
        self.mutable_edges = set()
        self.target_edges = set()

        # Initialize data structures for proposals
        self.complex_proposals = set()
        self.simple_proposals = set()
        self.xyz_to_edge = dict()
        self.kdtree = None

        # Initialize bounding box (if exists)
        self.bbox = img_bbox
        if self.bbox:
            self.origin = img_bbox["min"].astype(int)
            self.shape = (img_bbox["max"] - img_bbox["min"]).astype(int)
        else:
            self.origin = np.array([0, 0, 0], dtype=int)
            self.shape = None

    def copy_graph(self, add_attrs=False):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes(data=add_attrs))
        if add_attrs:
            for edge in self.get_edges_temp():
                i, j = tuple(edge)
                graph.add_edge(i, j, **self.get_edge_data(i, j))
        else:
            graph.add_edges_from(self.get_edges_temp())
        return graph

    def get_edges_temp(self):
        edges = []
        for edge in self.edges:
            edge = frozenset(edge)
            if edge not in self.mutable_edges:
                edges.append(edge)
        return edges

    def init_densegraph(self):
        self.densegraph = DenseGraph(self.swc_paths)

    # --- Add nodes or edges ---
    def add_component(self, irreducibles, swc_id):
        # Nodes
        node_ids = dict()
        cur_id = len(self.nodes) + 1
        node_ids, cur_id = self.__add_nodes(
            irreducibles, "leafs", node_ids, cur_id, swc_id
        )
        node_ids, cur_id = self.__add_nodes(
            irreducibles, "junctions", node_ids, cur_id, swc_id
        )

        # Add edges
        for edge, values in irreducibles["edges"].items():
            i, j = edge
            self.add_edge(
                node_ids[i],
                node_ids[j],
                radius=values["radius"],
                xyz=values["xyz"],
                swc_id=swc_id,
            )
            edge = (node_ids[i], node_ids[j])
            for xyz in values["xyz"][::2]:
                self.xyz_to_edge[tuple(xyz)] = edge
            self.xyz_to_edge[tuple(values["xyz"][-1])] = edge

    def __add_nodes(self, nodes, key, node_ids, cur_id, swc_id):
        for i in nodes[key].keys():
            node_ids[i] = cur_id
            self.add_node(
                node_ids[i],
                xyz=nodes[key][i]["xyz"],
                radius=nodes[key][i]["radius"],
                swc_id=swc_id,
            )
            if key == "leafs":
                self.leafs.add(cur_id)
            else:
                self.junctions.add(cur_id)
            cur_id += 1
        return node_ids, cur_id

    # --- Proposal Generation ---
    def generate_proposals(
        self,
        radius,
        n_proposals_per_leaf=3,
        optimize=False,
        optimization_depth=10,
    ):
        """
        Generates edges for the graph.

        Returns
        -------
        None

        """
        self.init_kdtree()
        self.mutable_edges = set()
        for leaf in self.leafs:
            if not self.is_contained(leaf):
                continue
            xyz_leaf = self.nodes[leaf]["xyz"]
            proposals = self.__get_proposals(
                leaf, xyz_leaf, n_proposals_per_leaf, radius
            )
            for xyz in proposals:
                # Extract info on mutable connection
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Get connecting node
                d1 = check_dists(xyz_leaf, xyz, self.nodes[i]["xyz"], radius)
                d2 = check_dists(xyz_leaf, xyz, self.nodes[j]["xyz"], radius)
                if d1 and self.is_contained(i):
                    xyz = deepcopy(self.nodes[i]["xyz"])
                    node = i
                elif d2 and self.is_contained(j):
                    xyz = deepcopy(self.nodes[j]["xyz"])
                    node = j
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.split_edge((i, j), attrs, idxs[0])

                # Add edge
                self.add_edge(leaf, node, xyz=np.array([xyz_leaf, xyz]))
                self.mutable_edges.add(frozenset((leaf, node)))

        # Check whether to optimization proposals
        if optimize:
            self.run_optimization()

    def __get_proposals(
        self, query_id, query_xyz, n_proposals_per_leaf, radius
    ):
        """
        Generates edge proposals for node "query_id" by finding points on
        distinct connected components near "query_xyz".

        Parameters
        ----------
        query_id : int
            Node id of the query node.
        query_xyz : tuple[float]
            (x,y,z) coordinates of the query node.
        n_proposals_per_leaf : int
            Number of proposals generated from node "query_id".
        radius : float
            Maximum Euclidean length of edge proposal.

        Returns
        -------
        list
            List of "n_proposals_per_leaf" best edge proposals generated from
            "query_node".

        """
        best_xyz = dict()
        best_dist = dict()
        query_swc_id = self.nodes[query_id]["swc_id"]
        for xyz in self._query_kdtree(query_xyz, radius):
            # Check whether xyz is contained
            if not self.is_contained(xyz, buffer=36):
                continue

            # Check whether
            edge = self.xyz_to_edge[tuple(xyz)]
            swc_id = gutils.get_edge_attr(self, edge, "swc_id")
            if swc_id != query_swc_id:
                d = get_dist(xyz, query_xyz)
                if swc_id not in best_dist.keys():
                    best_xyz[swc_id] = tuple(xyz)
                    best_dist[swc_id] = d
                elif d < best_dist[swc_id]:
                    best_xyz[swc_id] = tuple(xyz)
                    best_dist[swc_id] = d
        return self._get_best_edges(best_dist, best_xyz, n_proposals_per_leaf)

    def _get_best_edges(self, dists, xyz, n_proposals_per_leaf):
        """
        Gets the at most "n_proposals_per_leaf" nodes that are closest to
        "xyz".

        """
        if len(dists.keys()) > n_proposals_per_leaf:
            keys = sorted(dists, key=dists.__getitem__)
            return [xyz[key] for key in keys[0:n_proposals_per_leaf]]
        else:
            return list(xyz.values())

    def split_edge(self, edge, attrs, idx):
        """
        Splits "edge" into two distinct edges by making the subnode at "idx" a
        new node in "self".

        Parameters
        ----------
        edge : tuple
            Edge to be split.
        attrs : dict
            Attributes of "edge".
        idx : int
            Index of subnode that will become a new node in "self".

        Returns
        -------
        new_node : int
            Node ID of node that was created.

        """
        # Remove old edge
        (i, j) = edge
        self.remove_edge(i, j)

        # Add new node and split edge
        new_node = len(self.nodes) + 1
        self.add_node(
            new_node,
            xyz=tuple(attrs["xyz"][idx]),
            radius=attrs["radius"][idx],
            swc_id=attrs["swc_id"],
        )
        self.__add_edge((i, new_node), attrs, np.arange(0, idx + 1))
        self.__add_edge(
            (new_node, j), attrs, np.arange(idx, len(attrs["xyz"]))
        )
        return new_node

    def __add_edge(self, edge, attrs, idxs):
        self.add_edge(
            edge[0],
            edge[1],
            xyz=attrs["xyz"][idxs],
            radius=attrs["radius"][idxs],
            swc_id=attrs["swc_id"],
        )
        for xyz in attrs["xyz"][idxs]:
            self.xyz_to_edge[tuple(xyz)] = edge

    def init_kdtree(self):
        """
        Builds a KD-Tree from the (x,y,z) coordinates of the subnodes of
        each connected component in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        if not self.kdtree:
            self.kdtree = KDTree(list(self.xyz_to_edge.keys()))

    def _query_kdtree(self, query, d):
        """
        Parameters
        ----------
        query : int
            Node id.
        d : float
            Distance from query that is searched.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "d" from "query".

        """
        idxs = self.kdtree.query_ball_point(query, d, return_sorted=True)
        return self.kdtree.data[idxs]

    # --- Optimize Proposals ---
    def run_optimization(self):
        driver = "n5" if ".n5" in self.img_path else "zarr"
        img = utils.get_superchunk(
            self.img_path, driver, self.origin, self.shape, from_center=False
        )
        for edge in self.mutable_edges:
            xyz_1, xyz_2 = geometry.optimize_alignment(self, img, edge)
            self.edges[edge]["xyz"] = np.array([xyz_1, xyz_2])

    def get_branch(self, xyz_or_node, key="xyz"):
        if type(xyz_or_node) == int:
            nb = self.get_nbs_temp(xyz_or_node)[0]
            return self.orient_edge((xyz_or_node, nb), xyz_or_node, key=key)
        else:
            edge = self.xyz_to_edge[tuple(xyz_or_node)]
            branch = deepcopy(self.edges[edge]["xyz"])
            if not (branch[0] == xyz_or_node).all():
                return np.flip(branch, axis=0)
            else:
                return branch

    def get_branches(self, i, key="xyz"):
        branches = []
        for j in self.get_nbs_temp(i):
            branches.append(self.orient_edge((i, j), i, key=key))
        return branches

    def orient_edge(self, edge, i, key="xyz"):
        if (self.edges[edge]["xyz"][0, :] == self.nodes[i]["xyz"]).all():
            return self.edges[edge][key]
        else:
            return np.flip(self.edges[edge][key], axis=0)

    # --- Ground Truth Generation ---
    def init_targets(self, target_neurograph):
        # Initializations
        msg = "Provide swc_dir/swc_paths to initialize target edges!"
        assert target_neurograph.swc_paths, msg
        pred_graph = self.copy_graph()
        target_neurograph.init_densegraph()
        target_neurograph.init_kdtree()
        self.target_edges = set()

        # Add best simple edges
        remaining_proposals = []
        proposals = self.filter_infeasible(target_neurograph)
        dists = [self.compute_length(edge) for edge in proposals]
        for idx in np.argsort(dists):
            edge = proposals[idx]
            if self.is_simple(edge):
                add_bool = self.is_target(
                    target_neurograph,
                    pred_graph,
                    edge,
                    dist=7,
                    ratio=0.7,
                    exclude=10,
                )
                if add_bool:
                    pred_graph.add_edges_from([edge])
                    self.target_edges.add(edge)
                    continue
            remaining_proposals.append(edge)

        # Check remaining proposals
        dists = [self.compute_length(edge) for edge in remaining_proposals]
        for idx in np.argsort(dists):
            edge = remaining_proposals[idx]
            add_bool = self.is_target(
                target_neurograph,
                pred_graph,
                edge,
                dist=8,
                ratio=0.5,
                exclude=10,
            )
            if add_bool:
                pred_graph.add_edges_from([edge])
                self.target_edges.add(edge)

    def filter_infeasible(self, target_neurograph):
        proposals = list()
        for edge in self.mutable_edges:
            i, j = tuple(edge)
            xyz_i = self.nodes[i]["xyz"]
            xyz_j = self.nodes[j]["xyz"]
            if target_neurograph.is_feasible(xyz_i, xyz_j):
                proposals.append(edge)
        return proposals

    def is_feasible(self, xyz_1, xyz_2):
        # Check if edges are identical
        edge_1 = self.xyz_to_edge[self.get_projection(xyz_1)[0]]
        edge_2 = self.xyz_to_edge[self.get_projection(xyz_2)[0]]
        if edge_1 == edge_2:
            return True

        # Check if edges are adjacent
        i, j = tuple(edge_1)
        k, l = tuple(edge_2)
        bool_i = self.is_nb(i, k) or self.is_nb(i, l)
        bool_j = self.is_nb(j, k) or self.is_nb(j, l)
        if bool_i or bool_j:
            return True

        # Not feasible
        return False

    def is_target(
        self, target_graph, pred_graph, edge, dist=5, ratio=0.5, exclude=10
    ):
        # Check for cycle
        if gutils.creates_cycle(pred_graph, tuple(edge)):
            return False

        # Check projection distance
        xyz_i = self.edges[edge]["xyz"][0]
        xyz_j = self.edges[edge]["xyz"][-1]
        _, d_i = target_graph.get_projection(xyz_i)
        _, d_j = target_graph.get_projection(xyz_j)
        if d_i > dist or d_j > dist:
            return False

        # Check alignment
        aligned = target_graph.densegraph.is_aligned(
            xyz_i, xyz_j, ratio_threshold=ratio, exclude=exclude
        )
        return True if aligned else False

    # --- Generate reconstructions post-inference
    def get_reconstruction(self, proposals, upd_self=False):
        reconstruction = self.copy_graph(add_attrs=True)
        for edge in proposals:
            i, j = tuple(edge)
            r_i = self.nodes[i]["radius"]
            r_j = self.nodes[j]["radius"]
            reconstruction.add_edge(
                i, j, xyz=self.edges[i, j]["xyz"], radius=[r_i, r_j]
            )
        return reconstruction

    # --- Utils ---
    def n_nodes(self):
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

    def n_edges(self):
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

    def n_mutables(self):
        """
        Computes number of mutable edges in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of mutable edges in the graph.

        """
        return len(self.mutable_edges)

    def get_degree_temp(self, i):
        return len(self.get_nbs_temp(i))

    def get_nbs_temp(self, i):
        nbs = []
        for j in self.neighbors(i):
            if frozenset((i, j)) not in self.mutable_edges:
                nbs.append(j)
        return nbs

    def get_immutables_temp(self):
        return [
            edge
            for edge in self.edges
            if frozenset(edge) not in self.mutable_edges
        ]

    def compute_length(self, edge, metric="l2"):
        xyz_1, xyz_2 = self.get_edge_attr(edge, "xyz")
        return get_dist(xyz_1, xyz_2, metric=metric)

    def path_length(self, metric="l2"):
        length = 0
        for edge in self.edges:
            if edge not in self.mutable_edges:
                length += self.compute_length(edge, metric=metric)
        return length

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        proj_xyz = tuple(self.kdtree.data[idx])
        proj_dist = get_dist(proj_xyz, xyz)
        return proj_xyz, proj_dist

    def is_nb(self, i, j):
        return True if i in self.neighbors(j) else False

    def is_contained(self, node_or_xyz, buffer=0):
        if self.bbox:
            img_coord = self.to_img(node_or_xyz)
            return utils.is_contained(self.bbox, img_coord, buffer=buffer)
        else:
            return True

    def to_img(self, node_or_xyz, shift=False):
        shift = self.origin if shift else np.zeros((3))
        if type(node_or_xyz) == int:
            img_coord = utils.to_img(self.nodes[node_or_xyz]["xyz"])
        else:
            img_coord = utils.to_img(node_or_xyz)
        return img_coord - shift

    def is_leaf(self, i):
        return True if self.get_degree_temp(i) == 1 else False

    def get_edge_attr(self, edge, key):
        xyz_arr = gutils.get_edge_attr(self, edge, key)
        return xyz_arr[0], xyz_arr[-1]

    def get_complex_proposals(self):
        return set([e for e in self.mutable_edges if not self.is_simple(e)])

    def get_simple_proposals(self):
        return set([e for e in self.mutable_edges if self.is_simple(e)])

    def is_simple(self, edge):
        i, j = tuple(edge)
        return True if self.is_leaf(i) and self.is_leaf(j) else False

    def to_patch_coords(self, edge, midpoint, chunk_size):
        patch_coords = []
        for xyz in self.edges[edge]["xyz"]:
            img_coord = self.to_img(xyz)
            coord = utils.img_to_patch(img_coord, midpoint, chunk_size)
            patch_coords.append(coord)
        return patch_coords

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
