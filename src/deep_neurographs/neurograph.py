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
from deep_neurographs.groundtruth_generation import init_targets

SUPPORTED_LABEL_MASK_TYPES = [dict, np.array, ts.TensorStore]


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files.

    """

    def __init__(
        self,
        img_bbox=None,
        swc_paths=None,
        img_path=None,
        label_mask=None,
        train_model=False,
    ):
        super(NeuroGraph, self).__init__()
        # Initialize paths
        self.img_path = img_path
        self.label_mask = label_mask
        self.swc_paths = swc_paths
        self.swc_ids = set()

        # Initialize node and edge sets
        self.leafs = set()
        self.junctions = set()
        self.proposals = dict()
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
            if edge not in self.proposals:
                edges.append(edge)
        return edges

    def init_densegraph(self):
        return DenseGraph(self.swc_paths)

    # --- Add nodes or edges ---
    def add_swc_id(self, swc_id):
        self.swc_ids.add(swc_id)

    def add_component(self, irreducibles):
        # Nodes
        node_ids = dict()
        cur_id = len(self.nodes) + 1
        swc_id = irreducibles["swc_id"]
        node_ids, cur_id = self.__add_nodes(
            irreducibles, "leafs", node_ids, cur_id, swc_id
        )
        node_ids, cur_id = self.__add_nodes(
            irreducibles, "junctions", node_ids, cur_id, swc_id
        )
        self.add_swc_id(swc_id)

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
                proposals=set(),
                radius=nodes[key][i]["radius"],
                swc_id=swc_id,
                xyz=nodes[key][i]["xyz"],
            )
            if key == "leafs":
                self.leafs.add(cur_id)
            else:
                self.junctions.add(cur_id)
            cur_id += 1
        return node_ids, cur_id

    # --- Proposal and Ground Truth Generation ---
    def generate_proposals(
        self,
        radius,
        n_proposals_per_leaf=3,
        optimize=False,
        optimization_depth=10,
        restrict=True,
    ):
        """
        Generates edges for the graph.

        Returns
        -------
        None

        """
        self.init_kdtree()
        self.proposals = dict()
        existing_connections = (
            dict()
        )  # key: swc_id, values: {other swc_id: node}
        for leaf in self.leafs:
            if not self.is_contained(leaf, buffer=36):
                continue

            leaf_swc_id = self.nodes[leaf]["swc_id"]
            xyz_leaf = self.nodes[leaf]["xyz"]
            proposals = self.__get_proposals(
                leaf, xyz_leaf, n_proposals_per_leaf, radius
            )
            for xyz in proposals:
                # Extract info on proposal
                (i, j) = self.xyz_to_edge[xyz]
                attrs = self.get_edge_data(i, j)

                # Check for existing connection btw components
                swc_id = self.nodes[i]["swc_id"]
                if swc_id in existing_connections.keys() and restrict:
                    if leaf_swc_id in existing_connections[swc_id].keys():
                        edge = existing_connections[swc_id][leaf_swc_id]
                        len1 = self.node_xyz_dist(leaf, xyz)
                        len2 = self.proposal_length(edge)
                        if len1 < len2:
                            node1, node2 = tuple(edge)
                            self.nodes[node1]["proposals"].remove(node2)
                            self.nodes[node2]["proposals"].remove(node1)
                            del self.proposals[edge]
                        else:
                            continue

                # Check if proposal is connected to double
                """
                leaf_component = gutils.get_component(self, leaf)
                component = gutils.get_component(self, xyz)
                if geometry_utils.is_double(leaf_component, component, leaf):
                    continue
                """

                # Get connecting node
                d1 = check_dists(xyz_leaf, xyz, self.nodes[i]["xyz"], radius)
                d2 = check_dists(xyz_leaf, xyz, self.nodes[j]["xyz"], radius)
                if d1 and self.is_contained(i, buffer=36):
                    xyz = deepcopy(self.nodes[i]["xyz"])
                    node = i
                elif d2 and self.is_contained(j, buffer=36):
                    xyz = deepcopy(self.nodes[j]["xyz"])
                    node = j
                else:
                    idxs = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0]
                    node = self.split_edge((i, j), attrs, idxs[0])

                # Add edge
                edge = frozenset({leaf, node})
                self.proposals[edge] = {"xyz": np.array([xyz_leaf, xyz])}
                self.nodes[node]["proposals"].add(leaf)
                self.nodes[leaf]["proposals"].add(node)

                if leaf_swc_id in existing_connections.keys():
                    existing_connections[leaf_swc_id][swc_id] = edge
                else:
                    existing_connections[leaf_swc_id] = {swc_id: edge}

                if swc_id in existing_connections.keys():
                    existing_connections[swc_id][leaf_swc_id] = edge
                else:
                    existing_connections[swc_id] = {leaf_swc_id: edge}

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
        for xyz in self.query_kdtree(query_xyz, radius):
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
        return self.get_best_proposals(
            best_dist, best_xyz, n_proposals_per_leaf
        )

    def get_best_proposals(self, dists, xyz, n_proposals_per_leaf):
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
            proposals=set(),
            radius=attrs["radius"][idx],
            swc_id=attrs["swc_id"],
            xyz=tuple(attrs["xyz"][idx]),
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

    def init_targets(self, target_neurograph):
        target_neurograph.init_kdtree()
        self.target_edges = init_targets(target_neurograph, self)

    # --- Optimize Proposals ---
    def run_optimization(self):
        driver = "n5" if ".n5" in self.img_path else "zarr"
        img = utils.get_superchunk(
            self.img_path, driver, self.origin, self.shape, from_center=False
        )
        for edge in self.proposals:
            xyz_1, xyz_2 = geometry.optimize_alignment(self, img, edge)
            self.proposals[edge]["xyz"] = np.array([xyz_1, xyz_2])

    def get_branches(self, i, key="xyz"):
        branches = []
        for j in self.neighbors(i):
            branches.append(self.orient_edge((i, j), i, key=key))
        return branches

    def orient_edge(self, edge, i, key="xyz"):
        if (self.edges[edge][key][0] == self.nodes[i][key]).all():
            return self.edges[edge][key]
        else:
            return np.flip(self.edges[edge][key], axis=0)

    # -- kdtree --
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

    def query_kdtree(self, xyz, d):
        """
        Parameters
        ----------
        xyz : int
            Node id.
        d : float
            Distance from "xyz" that is searched.

        Returns
        -------
        generator[tuple]
            Generator that generates the xyz coordinates cooresponding to all
            nodes within a distance of "d" from "xyz".

        """
        idxs = self.kdtree.query_ball_point(xyz, d, return_sorted=True)
        return self.kdtree.data[idxs]

    def get_projection(self, xyz):
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])

    # --- utils ---
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

    def n_proposals(self):
        """
        Computes number of edges proposals in the graph.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of edge proposals in the graph.

        """
        return len(self.proposals)

    def get_proposals(self):
        return list(self.proposals.keys())

    def remove_proposal(self, edge):
        del self.proposals[edge]

    def proposal_xyz(self, edge):
        return tuple(self.proposals[edge]["xyz"])

    def proposal_length(self, edge):
        i, j = tuple(edge)
        return get_dist(self.nodes[i]["xyz"], self.nodes[j]["xyz"])

    def node_xyz_dist(self, node, xyz):
        return get_dist(xyz, self.nodes[node]["xyz"])

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
        return True if self.degree[i] == 1 else False

    def get_edge_attr(self, edge, key):
        xyz_arr = gutils.get_edge_attr(self, edge, key)
        return xyz_arr[0], xyz_arr[-1]

    def get_complex_proposals(self):
        return set([e for e in self.get_proposals() if not self.is_simple(e)])

    def get_simple_proposals(self):
        return set([e for e in self.get_proposals() if self.is_simple(e)])

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

    def get_reconstruction(self, proposals, upd_self=False):
        reconstruction = self.copy_graph(add_attrs=True)
        for edge in proposals:
            i, j = tuple(edge)
            r_i = self.nodes[i]["radius"]
            r_j = self.nodes[j]["radius"]
            reconstruction.add_edge(
                i, j, xyz=self.proposals[i, j]["xyz"], radius=[r_i, r_j]
            )
        return reconstruction
