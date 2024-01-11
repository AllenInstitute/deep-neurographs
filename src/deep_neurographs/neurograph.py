"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "NeuroGraph".

"""

from copy import deepcopy
from time import time

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree

from deep_neurographs import geometry_utils
from deep_neurographs import graph_utils as gutils
from deep_neurographs import swc_utils, utils
from deep_neurographs.densegraph import DenseGraph
from deep_neurographs.geometry_utils import dist as get_dist

SUPPORTED_LABEL_MASK_TYPES = [dict, np.array, ts.TensorStore]


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files. This type of graph has two sets of edges referred
    to as "mutable" and "immutable".

    """

    def __init__(
        self,
        swc_dir=None,
        img_path=None,
        label_mask=None,
        optimize_depth=10,
        optimize_alignment=False,
        optimize_path=False,
        origin=None,
        shape=None,
        size_threshold=30,
    ):
        """
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        super(NeuroGraph, self).__init__()
        self.path = swc_dir
        self.label_mask = label_mask
        self.leafs = set()
        self.junctions = set()
        self.size_threshold = size_threshold

        self.immutable_edges = set()
        self.mutable_edges = set()
        self.target_edges = set()
        self.xyz_to_edge = dict()
        self.kdtree = None

        self.img_path = img_path
        self.optimize_depth = optimize_depth
        self.optimize_alignment = optimize_alignment
        self.optimize_path = optimize_path
        self.simple_proposals = set()
        self.complex_proposals = set()

        self.bbox = None
        self.shape = shape
        if origin and shape:
            self.bbox = {
                "min": np.array(origin),
                "max": np.array([origin[i] + shape[i] for i in range(3)]),
            }
            self.origin = np.array(origin)

    def init_immutable_graph(self, add_attrs=False):
        immutable_graph = nx.Graph()
        immutable_graph.add_nodes_from(self.nodes(data=add_attrs))
        if add_attrs:
            for edge in self.immutable_edges:
                i, j = tuple(edge)
                immutable_graph.add_edge(i, j, **self.get_edge_data(i, j))
        else:
            immutable_graph.add_edges_from(self.immutable_edges)
        return immutable_graph

    def init_predicted_graph(self):
        self.predicted_graph = self.init_immutable_graph()

    def init_densegraph(self):
        self.densegraph = DenseGraph(self.path)

    # --- Add nodes or edges ---
    def ingest_swc_from_local(
        self, path, prune=True, prune_depth=16, smooth=True
    ):
        # Parse swc
        swc_id = utils.get_id(path)
        swc_dict = swc_utils.parse_local_swc(path, bbox=self.bbox)
        if len(swc_dict["xyz"]) > self.size_threshold:
            return None

        # Build neurograph
        t0 = time()
        leafs, junctions, edges = gutils.get_irreducibles(
            swc_dict, prune=prune, depth=prune_depth, smooth=smooth
        )
        self.extraction += time() - t0
        self.add_immutables(swc_id, swc_dict, leafs, junctions, edges)

    def add_immutables(self, swc_id, swc_dict, leafs, junctions, edges):
        """
        Adds nodes to graph from a dictionary generated from an swc files.

        Parameters
        ----------
        node_id : int
            Node id.
        swc_dict : dict
            Dictionary generated from an swc where the keys are swc
            attributes.

        Returns
        -------
        None.

        """
        # Add nodes

        node_id = dict()
        for i in list(leafs) + list(junctions):
            node_id[i] = len(self.nodes)
            self.add_node(
                node_id[i],
                xyz=np.array(swc_dict["xyz"][i]),
                radius=swc_dict["radius"][i],
                swc_id=swc_id,
            )

        # Add edges
        t0 = time()
        for i, j in edges.keys():
            # Get edge
            edge = (node_id[i], node_id[j])
            xyz = np.array(edges[(i, j)]["xyz"])
            radii = np.array(edges[(i, j)]["radius"])

            # Add edge
            self.immutable_edges.add(frozenset(edge))
            self.add_edge(
                node_id[i], node_id[j], xyz=xyz, radius=radii, swc_id=swc_id
            )
            xyz_to_edge = dict((tuple(xyz), edge) for xyz in xyz)
            check_xyz = set(xyz_to_edge.keys())
            collisions = check_xyz.intersection(set(self.xyz_to_edge.keys()))
            if len(collisions) > 0:
                for xyz in collisions:
                    del xyz_to_edge[xyz]
            self.xyz_to_edge.update(xyz_to_edge)
        self.add_edges_timer += time() - t0

        # Update leafs and junctions
        for l in leafs:
            self.leafs.add(node_id[l])

        for j in junctions:
            self.junctions.add(node_id[j])

    # --- Proposal Generation ---
    def generate_proposals(self, n_proposals_per_leaf=3, search_radius=25.0):
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
                leaf, xyz_leaf, n_proposals_per_leaf, search_radius
            )
            for xyz in proposals:
                # Extract info on mutable connection
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Get connecting node
                contained_j = self.is_contained(j)
                if get_dist(xyz, attrs["xyz"][0]) < 10 and self.is_contained(
                    i
                ):
                    node = i
                    xyz = self.nodes[node]["xyz"]
                elif get_dist(xyz, attrs["xyz"][-1]) < 10 and contained_j:
                    node = j
                    xyz = self.nodes[node]["xyz"]
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.add_immutable_node((i, j), attrs, idxs[0])

                # Add edge
                self.add_edge(leaf, node, xyz=np.array([xyz_leaf, xyz]))
                self.mutable_edges.add(frozenset((leaf, node)))

        if self.optimize_alignment or self.optimize_path:
            self.run_optimization()

    def __get_proposals(
        self, query_id, query_xyz, n_proposals_per_leaf, search_radius
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
        search_radius : float
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
        for xyz in self._query_kdtree(query_xyz, search_radius):
            if not self.is_contained(xyz):
                continue
            xyz = tuple(xyz)
            edge = self.xyz_to_edge[xyz]
            swc_id = gutils.get_edge_attr(self, edge, "swc_id")
            if swc_id != query_swc_id:
                d = get_dist(xyz, query_xyz)
                if swc_id not in best_dist.keys():
                    best_xyz[swc_id] = xyz
                    best_dist[swc_id] = d
                elif d < best_dist[swc_id]:
                    best_xyz[swc_id] = xyz
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
        self.__add_edge((i, node_id), attrs, np.arange(0, idx + 1))
        self.__add_edge((node_id, j), attrs, np.arange(idx, len(attrs["xyz"])))
        return node_id

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
        self.immutable_edges.add(frozenset(edge))

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
        origin = utils.apply_anisotropy(self.origin, return_int=True)
        img = utils.get_superchunk(
            self.img_path, "zarr", origin, self.shape, from_center=False
        )
        for edge in self.mutable_edges:
            xyz_1, xyz_2 = geometry_utils.optimize_alignment(self, img, edge)
            proposal = [self.to_world(xyz_1)]
            if self.optimize_path:
                path = geometry_utils.optimize_path(
                    img, self.origin, xyz_1, xyz_2
                )
                proposal.append(path)
            proposal.append(self.to_world(xyz_2))
            self.edges[edge]["xyz"] = np.vstack(proposal)

    def get_branch(self, xyz_or_node):
        if type(xyz_or_node) == int:
            nb = self.get_immutable_nbs(xyz_or_node)[0]
            return self.orient_edge((xyz_or_node, nb), xyz_or_node)
        else:
            edge = self.xyz_to_edge[tuple(xyz_or_node)]
            branch = deepcopy(self.edges[edge]["xyz"])
            if not (branch[0] == xyz_or_node).all():
                return np.flip(branch, axis=0)
            else:
                return branch

    def get_branches(self, i):
        branches = []
        for j in self.neighbors(i):
            if frozenset((i, j)) in self.immutable_edges:
                branches.append(self.orient_edge((i, j), i))
        return branches

    def orient_edge(self, edge, i):
        if (self.edges[edge]["xyz"][0, :] == self.nodes[i]["xyz"]).all():
            return self.edges[edge]["xyz"]
        else:
            return np.flip(self.edges[edge]["xyz"], axis=0)

    # --- Ground Truth Generation ---
    def init_targets(self, target_neurograph):
        # Initializations
        msg = "Error: Provide swc_dir/swc_paths to initialize target edges!"
        assert target_neurograph.path, msg
        target_neurograph.init_densegraph()
        target_neurograph.init_kdtree()
        self.target_edges = set()
        self.init_predicted_graph()

        # Add best simple edges
        remaining_proposals = []
        proposals = self.filter_infeasible(target_neurograph)
        dists = [self.compute_length(edge) for edge in proposals]
        for idx in np.argsort(dists):
            edge = proposals[idx]
            if self.is_simple(edge):
                add_bool = self.is_target(
                    target_neurograph, edge, dist=5, ratio=0.7, exclude=10
                )
                if add_bool:
                    self.target_edges.add(edge)
                    continue
            remaining_proposals.append(edge)

        # Check remaining proposals
        dists = [self.compute_length(edge) for edge in remaining_proposals]
        for idx in np.argsort(dists):
            edge = remaining_proposals[idx]
            add_bool = self.is_target(
                target_neurograph, edge, dist=7, ratio=0.4, exclude=15
            )
            if add_bool:
                self.target_edges.add(edge)

        # Print results
        # target_ratio = len(self.target_edges) / len(self.mutable_edges)
        # print("# target edges:", len(self.target_edges))
        # print("% target edges in mutable:", target_ratio)
        # print("")

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
        self, target_graph, edge, dist=5, ratio=0.5, strict=True, exclude=10
    ):
        # Check for cycle
        i, j = tuple(edge)
        if self.creates_cycle((i, j)):
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
        reconstruction = self.init_immutable_graph(add_attrs=True)
        for edge in proposals:
            i, j = tuple(edge)
            r_i = self.nodes[i]["radius"]
            r_j = self.nodes[j]["radius"]
            reconstruction.add_edge(
                i, j, xyz=self.edges[i, j]["xyz"], radius=[r_i, r_j]
            )
        return reconstruction

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

    def num_immutables(self):
        """
        Computes number of immutable edges in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of immutable edges in the graph.

        """
        return len(self.immutable_edges)

    def num_mutables(self):
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

    def immutable_degree(self, i):
        degree = 0
        for j in self.neighbors(i):
            if frozenset((i, j)) in self.immutable_edges:
                degree += 1
        return degree

    def get_immutable_nbs(self, i):
        nbs = []
        for j in self.neighbors(i):
            if frozenset((i, j)) in self.immutable_edges:
                nbs.append(j)
        return nbs

    def compute_length(self, edge, metric="l2"):
        i, j = tuple(edge)
        xyz_1, xyz_2 = self.get_edge_attr("xyz", i, j)
        return get_dist(xyz_1, xyz_2, metric=metric)

    def path_length(self, metric="l2"):
        length = 0
        for edge in self.immutable_edges:
            length += self.compute_length(edge, metric=metric)
        return length

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        proj_xyz = tuple(self.kdtree.data[idx])
        proj_dist = get_dist(proj_xyz, xyz)
        return proj_xyz, proj_dist

    def is_nb(self, i, j):
        return True if i in self.neighbors(j) else False

    def is_contained(self, node_or_xyz):
        if self.bbox:
            if type(node_or_xyz) == int:
                node_or_xyz = deepcopy(self.nodes[node_or_xyz]["xyz"])
            xyz = utils.apply_anisotropy(node_or_xyz - self.bbox["min"])
            img_shape = np.array(self.shape)
            for i in range(3):
                lower_bool = xyz[i] < 32
                upper_bool = xyz[i] > img_shape[i] - 32
                if lower_bool or upper_bool:
                    return False
        return True

    def is_leaf(self, i):
        return True if self.immutable_degree(i) == 1 else False

    def creates_cycle(self, edge):
        self.predicted_graph.add_edges_from([edge])
        try:
            nx.find_cycle(self.predicted_graph)
        except:
            return False
        self.predicted_graph.remove_edges_from([edge])
        return True

    def get_edge_attr(self, key, i, j):
        attr_1 = self.nodes[i][key]
        attr_2 = self.nodes[j][key]
        return attr_1, attr_2

    def get_center(self):
        return geometry_utils.get_midpoint(self.bbox["min"], self.bbox["max"])

    def get_complex_proposals(self):
        return set([e for e in self.mutable_edges if not self.is_simple(e)])

    def get_simple_proposals(self):
        return set([e for e in self.mutable_edges if self.is_simple(e)])

    def is_simple(self, edge):
        i, j = tuple(edge)
        if self.immutable_degree(i) == 1 and self.immutable_degree(j) == 1:
            return True
        else:
            return False

    def to_img(self, node_or_xyz):
        if type(node_or_xyz) == int:
            node_or_xyz = deepcopy(self.nodes[node_or_xyz]["xyz"])
        return utils.to_img(node_or_xyz, shift=self.origin)

    def to_world(self, node_or_xyz, shift=[0, 0, 0]):
        if type(node_or_xyz) == int:
            node_or_xyz = deepcopy(self.nodes[node_or_xyz]["xyz"])
        return utils.to_world(node_or_xyz, shift=-self.origin)

    def to_patch_coords(self, edge, midpoint, chunk_size):
        patch_coords = []
        for xyz in self.edges[edge]["xyz"]:
            coord = utils.img_to_patch(self.to_img(xyz), midpoint, chunk_size)
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
