"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "FragmentsGraph".

NOTE: SAVE LABEL UPDATES --- THERE IS A BUG IN FEATURE GENERATION

"""
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import StringIO

import networkx as nx
import numpy as np
from numpy import concatenate
from scipy.spatial import KDTree

from deep_neurographs import generate_proposals, geometry
from deep_neurographs.geometry import dist as get_dist
from deep_neurographs.groundtruth_generation import init_targets
from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import img_util, swc_util, util


class NeuroGraph(nx.Graph):
    """
    A class of graphs whose nodes correspond to irreducible nodes from the
    predicted swc files.

    """

    def __init__(self, img_bbox=None, node_spacing=1):
        """
        Initializes an instance of NeuroGraph.

        Parameters
        ----------
        img_bbox : dict or None, optional
            Dictionary with the keys "min" and "max" which specify a bounding
            box in an image. The default is None.
        node_spacing : int, optional
            Spacing (in microns) between nodes. The default is 1.

        Returns
        -------
        None

        """
        super(NeuroGraph, self).__init__()
        # General class attributes
        self.leaf_kdtree = None
        self.node_cnt = 0
        self.node_spacing = node_spacing
        self.proposals = set()

        self.merged_ids = set()
        self.soma_ids = dict()
        self.swc_ids = set()
        self.xyz_to_edge = dict()

        # Bounding box (if applicable)
        self.bbox = img_bbox
        if self.bbox:
            self.origin = img_bbox["min"].astype(int)
            self.shape = (img_bbox["max"] - img_bbox["min"]).astype(int)
        else:
            self.origin = np.array([0, 0, 0], dtype=int)
            self.shape = None

    def copy_graph(self, add_attrs=False):
        graph = nx.Graph()
        nodes = deepcopy(self.nodes(data=add_attrs))
        graph.add_nodes_from(nodes)
        if add_attrs:
            for edge in self.edges:
                i, j = tuple(edge)
                graph.add_edge(i, j, **self.get_edge_data(i, j))
        else:
            graph.add_edges_from(deepcopy(self.edges))
        return graph

    def set_proxy_soma_ids(self, k):
        """
        Sets class attribute called "self.soma_swc_ids" which stores the swc
        ids of the "k" largest components. These components are used as a proxy
        for soma locations.

        Paramters
        ---------
        k : int
            Number of largest components to be set as proxy soma locations.

        Returns
        -------
        None

        """
        for i in gutil.largest_components(self, k):
            self.soma_ids[self.nodes[i]["swc_id"]] = i

    def get_leafs(self):
        """
        Gets all leaf nodes in graph.

        Parameters
        ----------
        None

        Returns
        -------
        list[int]
            Leaf nodes in graph.

        """
        return [i for i in self.nodes if self.is_leaf(i)]

    # --- Edit Graph --
    def add_component(self, irreducibles):
        """
        Adds a connected component to "self".

        Parameters
        ----------
        irreducibles : dict
            Dictionary containing the irreducibles of some connected component
            being added to "self". This dictionary must contain the keys:
            'leaf', 'branching', 'edge', and 'swc_id'.

        Returns
        -------
        None

        """
        swc_id = irreducibles["swc_id"]
        if swc_id not in self.swc_ids:
            # Nodes
            self.swc_ids.add(swc_id)
            ids = self.__add_nodes(irreducibles, "leaf", dict())
            ids = self.__add_nodes(irreducibles, "branching", ids)

            # Edges
            for (i, j), attrs in irreducibles["edge"].items():
                edge = (ids[i], ids[j])
                idxs = util.spaced_idxs(attrs["radius"], self.node_spacing)
                for key in ["radius", "xyz"]:
                    attrs[key] = attrs[key][idxs]
                self.__add_edge(edge, attrs, swc_id)

    def __add_nodes(self, irreducibles, node_type, node_ids):
        """
        Adds a set of "node_type" nodes from "irreducibles" to "self".

        Parameters
        ----------
        irreducibles : dict
            Dictionary containing the irreducibles of some connected component
            being added to "self".
        node_type : str
            Type of node being added to "self". This value must be either
            'leaf' or 'branching'.
        node_ids : dict
            Dictionary containing conversion from a node id in "irreducibles"
            to the corresponding node id in "self".

        Returns
        -------
        node_ids : dict
            Updated with corresponding node ids that were added in for loop.

        """
        for i in irreducibles[node_type].keys():
            cur_id = self.node_cnt + 1
            self.add_node(
                cur_id,
                proposals=set(),
                radius=irreducibles[node_type][i]["radius"],
                swc_id=irreducibles["swc_id"],
                xyz=irreducibles[node_type][i]["xyz"],
            )
            self.node_cnt += 1
            node_ids[i] = cur_id
        return node_ids

    def __add_edge(self, edge, attrs, swc_id):
        """
        Adds an edge to "self".

        Parameters
        ----------
        edge : tuple
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an swc file.
        swc_id : str
            swc id corresponding to edge.

        Returns
        -------
        None

        """
        i, j = tuple(edge)
        self.add_edge(
            i,
            j,
            length=attrs["length"],
            radius=attrs["radius"],
            xyz=attrs["xyz"],
            swc_id=swc_id,
        )
        self.xyz_to_edge.update({tuple(xyz): edge for xyz in attrs["xyz"]})

    def absorb_reducibles(self):
        """
        Absorbs reducible nodes (i.e. nodes with degree 2) in the graph by
        merging them into their neighboring nodes.

        Parameters
        ----------
        None

        Returns
        -------
        """
        nodes = deepcopy(self.nodes)
        for i in nodes:
            nbs = list(self.neighbors(i))
            if len(nbs) == 2 and len(self.nodes[i]["proposals"]) == 0:
                # Concatenate attributes
                len_1 = self.edges[i, nbs[0]]["length"]
                len_2 = self.edges[i, nbs[1]]["length"]
                xyz = self.branches(i, key="xyz")
                radius = self.branches(i, key="radius")
                attrs = {
                    "length": len_1 + len_2,
                    "radius": concatenate([np.flip(radius[0]), radius[1]]),
                    "xyz": concatenate([np.flip(xyz[0], axis=0), xyz[1]]),
                }

                # Update graph
                self.__add_edge(nbs, attrs, self.nodes[i]["swc_id"])
                self.remove_node(i)

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

        # Create node
        node_id = self.node_cnt + 1
        swc_id = attrs["swc_id"]
        self.add_node(
            node_id,
            proposals=set(),
            radius=attrs["radius"][idx],
            swc_id=swc_id,
            xyz=tuple(attrs["xyz"][idx]),
        )
        self.node_cnt += 1

        # Create edges
        n = len(attrs["xyz"])
        attrs_1 = {k: v[np.arange(idx + 1)] for k, v in attrs.items()}
        attrs_2 = {k: v[np.arange(idx, n)] for k, v in attrs.items()}
        self.__add_edge((i, node_id), attrs_1, swc_id)
        self.__add_edge((node_id, j), attrs_2, swc_id)
        return node_id

    # --- Proposal Generation ---
    def generate_proposals(
        self,
        search_radius,
        complex_bool=False,
        groundtruth_graph=None,
        long_range_bool=False,
        progress_bar=True,
        proposals_per_leaf=3,
        return_trimmed_proposals=False,
        trim_endpoints_bool=False,
    ):
        """
        Generates proposals from leaf nodes.

        Parameters
        ----------
        search_radius : float
            Search radius used to generate proposals.
        complex_bool : bool, optional
            Indication of whether to generate complex proposals. The default
            is False.
        groundtruth_graph : networkx.Graph, optional
            Ground truth graph. The default is None.
        long_range_bool : bool, optional
            Indication of whether to generate long range proposals. The
            default is False.
        progress_bar : bool, optional
            Indication of whether to print out a progress bar while generating
            proposals. The default is True.
        proposals_per_leaf : int, optional
            Maximum number of proposals generated for each leaf. The default
            is 3.
        return_trimmed_proposals, optional
            Indication of whether to return trimmed proposal ids. The default
            is False.
        trim_endpoints_bool : bool, optional
            Indication of whether to trim endpoints. The default is False.

        Returns
        -------
        None

        """
        # Main
        self.reset_proposals()
        self.set_proposals_per_leaf(proposals_per_leaf)
        n_trimmed = generate_proposals.run(
            self,
            search_radius,
            complex_bool=complex_bool,
            long_range_bool=long_range_bool,
            progress_bar=progress_bar,
            trim_endpoints_bool=trim_endpoints_bool,
        )

        # Establish groundtruth
        if groundtruth_graph:
            self.gt_accepts = init_targets(self, groundtruth_graph)
        else:
            self.gt_accepts = set()
        return n_trimmed

    def reset_proposals(self):
        """
        Deletes all previously generated proposals.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.proposals = set()
        self.xyz_to_proposal = dict()
        for i in self.nodes:
            self.nodes[i]["proposals"] = set()

    def set_proposals_per_leaf(self, proposals_per_leaf):
        """
        Sets the maximum number of proposals per leaf as a class attribute.

        Parameters
        ----------
        proposals_per_leaf : int
            Maximum number of proposals per leaf.

        Returns
        -------
        None

        """
        self.proposals_per_leaf = proposals_per_leaf

    def add_proposal(self, i, j):
        """
        Adds proposal between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node id.
        j : int
            Node id

        Returns
        -------
        None

        """
        proposal = frozenset((i, j))
        self.nodes[i]["proposals"].add(j)
        self.nodes[j]["proposals"].add(i)
        self.xyz_to_proposal[tuple(self.nodes[i]["xyz"])] = proposal
        self.xyz_to_proposal[tuple(self.nodes[j]["xyz"])] = proposal
        self.proposals.add(proposal)

    def remove_proposal(self, proposal):
        """
        Removes an existing proposal between two nodes.

        Parameters
        ----------
        proposal : frozenset
            Pair of node ids corresponding to a proposal.

        Returns
        -------
        None

        """
        i, j = tuple(proposal)
        self.nodes[i]["proposals"].remove(j)
        self.nodes[j]["proposals"].remove(i)
        self.proposals.remove(proposal)

    def is_single_proposal(self, proposal):
        """
        Determines whether "proposal" is the only proposal generated for the
        corresponding nodes.

        Parameters
        ----------
        proposal : frozenset
            Pair of node ids corresponding to a proposal.

        Returns
        -------
        bool
            Indiciation of "proposal" is the only proposal generated for the
        corresponding nodes.

        """
        i, j = tuple(proposal)
        single_i = len(self.nodes[i]["proposals"]) == 1
        single_j = len(self.nodes[j]["proposals"]) == 1
        return single_i and single_j

    def is_valid_proposal(self, leaf, i, complex_bool):
        """
        Determines whether a pair of nodes would form a "valid" proposal. A
        proposal is said to be valid if (1) "leaf" and "i" do not have swc_ids
        contained in "self.soma_ids" and (2) "i" is a leaf if "complex_bool"
        is False.

        Parmeters
        ---------
        leaf : int
            Leaf node id.
        i : int
            node id.
        complex_bool : bool
            Indication of whether complex proposals are should be generated.

        Returns
        -------
        bool
            Indication of whether proposal is valid.

        """
        if i is not None:
            skip_soma = self.is_soma(i) and self.is_soma(leaf)
            skip_complex = self.degree[i] > 1 and not complex_bool
            return not (skip_soma or skip_complex)
        else:
            return False

    def list_proposals(self):
        """
        Lists proposals in self.

        Parameters
        ----------
        None

        Returns
        -------
        list
            Proposals.

        """
        return list(self.proposals)

    # -- KDTree --
    def init_kdtree(self, node_type):
        """
        Builds a KD-Tree from the xyz coordinates of the subset of nodes
        indicated by "node_type".

        Parameters
        ----------
        node_type : None or str
            Type of node used to build kdtree.

        Returns
        -------
        None

        """
        assert node_type in ["leaf", "proposal"]
        if node_type == "leaf":
            self.leaf_kdtree = self.get_kdtree(node_type="leaf")
        elif node_type == "proposal":
            self.proposal_kdtree = self.get_kdtree(node_type="proposal")

    def get_kdtree(self, node_type=None):
        """
        Builds KD-Tree from xyz coordinates across all nodes and edges.

        Parameters
        ----------
        node_type : None or str, optional
            Type of nodes used to build kdtree.

        Returns
        -------
        KDTree
            KD-Tree generated from xyz coordinates across all nodes and edges.

        """
        # Get xyz coordinates
        if node_type == "leaf":
            xyz_list = [self.nodes[i]["xyz"] for i in self.get_leafs()]
        elif node_type == "proposal":
            xyz_list = list(self.xyz_to_proposal.keys())
        else:
            xyz_list = list(self.xyz_to_edge.keys())
        return KDTree(xyz_list)

    def query_kdtree(self, xyz, d, node_type):
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
        assert node_type in ["leaf", "proposal"]
        if node_type == "leaf":
            return geometry.query_ball(self.leaf_kdtree, xyz, d)
        elif node_type == "proposal":
            return geometry.query_ball(self.proposal_kdtree, xyz, d)

    # --- Proposal util ---
    def n_proposals(self):
        """
        Counts the number of proposals.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of proposals in the graph.

        """
        return len(self.proposals)

    def is_simple(self, proposal):
        i, j = tuple(proposal)
        return True if self.is_leaf(i) and self.is_leaf(j) else False

    def simple_proposals(self):
        return set([p for p in self.proposals if self.is_simple(p)])

    def complex_proposals(self):
        return set([p for p in self.proposals if not self.is_simple(p)])

    def proposal_length(self, proposal):
        return self.dist(*tuple(proposal))

    def proposal_midpoint(self, proposal):
        i, j = tuple(proposal)
        return geometry.midpoint(self.nodes[i]["xyz"], self.nodes[j]["xyz"])

    def proposal_radii(self, proposal):
        """
        Gets the radii of the nodes that comprise "proposal".

        Parameters
        ----------
        proposal : frozenset
            Pair of nodes that form a proposal.

        Returns
        -------
        numpy.ndarray
            radii of nodes that comprise "proposal".

        """
        i, j = tuple(proposal)
        return np.array([self.nodes[i]["radius"], self.nodes[j]["radius"]])

    def proposal_avg_radii(self, proposal):
        i, j = tuple(proposal)
        radii_i = self.branches(i, key="radius")
        radii_j = self.branches(j, key="radius")
        return np.array([avg_radius(radii_i), avg_radius(radii_j)])

    def proposal_xyz(self, proposal):
        """
        Gets the xyz coordinates of the nodes that comprise "proposal".

        Parameters
        ----------
        proposal : frozenset
            Pair of nodes that form a proposal.

        Returns
        -------
        numpy.ndarray
            xyz coordinates of nodes that comprise "proposal".

        """
        i, j = tuple(proposal)
        return np.array([self.nodes[i]["xyz"], self.nodes[j]["xyz"]])

    def proposal_labels(self, proposal):
        """
        Gets the xyz coordinates of the nodes that comprise "proposal".

        Parameters
        ----------
        proposal : frozenset
            Pair of nodes that form a proposal.

        Returns
        -------
        numpy.ndarray
            xyz coordinates of nodes that comprise "proposal".

        """
        i, j = tuple(proposal)
        return [int(self.nodes[i]["swc_id"]), int(self.nodes[j]["swc_id"])]

    def proposal_directionals(self, proposal, depth):
        # Extract branches
        i, j = tuple(proposal)
        branches_i = [geometry.truncate_path(b, depth) for b in self.branches(i)]
        branches_j = [geometry.truncate_path(b, depth) for b in self.branches(j)]
        origin = self.proposal_midpoint(proposal)

        # Compute tangent vectors
        direction_i = geometry.get_directional(branches_i, origin, depth)
        direction_j = geometry.get_directional(branches_j, origin, depth)
        direction = geometry.tangent(self.proposal_xyz(proposal))

        # Compute features
        inner_product_1 = abs(np.dot(direction, direction_i))
        inner_product_2 = abs(np.dot(direction, direction_j))
        if self.is_simple(proposal):
            inner_product_3 = np.dot(direction_i, direction_j)
        else:
            inner_product_3 = np.dot(direction_i, direction_j)
            if not self.is_simple(proposal):
                inner_product_3 = max(inner_product_3, -inner_product_3)
        return np.array([inner_product_1, inner_product_2, inner_product_3])

    def merge_proposal(self, proposal):
        i, j = tuple(proposal)
        somas_check = not (self.is_soma(i) and self.is_soma(j))
        if somas_check and self.check_proposal_degrees(i, j):
            # Dense attributes
            attrs = dict()
            self.nodes[i]["radius"] = 7.3141592
            self.nodes[j]["radius"] = 7.3141592
            for k in ["xyz", "radius"]:
                combine = np.vstack if k == "xyz" else np.array
                attrs[k] = combine([self.nodes[i][k], self.nodes[j][k]])

            # Sparse attributes
            if self.degree[i] == 1 and self.degree[j] == 1:
                e_i = (i, self.leaf_neighbor(i))
                e_j = (j, self.leaf_neighbor(j))
                len_ij = self.edges[e_i]["length"] + self.edges[e_j]["length"]
                attrs["length"] = len_ij
            elif self.degree[i] == 1:
                e_i = (i, self.leaf_neighbor(i))
                attrs["length"] = self.edges[e_i]["length"]
            else:
                e_j = (j, self.leaf_neighbor(j))
                attrs["length"] = self.edges[e_j]["length"]

            swc_id_i = self.nodes[i]["swc_id"]
            swc_id_j = self.nodes[j]["swc_id"]
            swc_id = swc_id_i if self.is_soma(i) else swc_id_j

            # Update graph
            self.merged_ids.add((swc_id_i, swc_id_j))
            self.upd_ids(swc_id, j if swc_id == swc_id_i else i)
            self.__add_edge((i, j), attrs, swc_id)
            self.proposals.remove(proposal)

    def check_proposal_degrees(self, i, j):
        one_leaf = self.degree[i] == 1 or self.degree[j] == 1
        branching = self.degree[i] > 2 or self.degree[j] > 2
        return one_leaf and not branching

    def upd_ids(self, swc_id, r):
        """
        Updates the swc_id of all nodes connected to "r".

        Parameters
        ----------
        swc_id : str
            Segment id.
        r : int
            Node.

        Returns
        -------
        None

        """
        queue = [r]
        visited = set()
        while len(queue) > 0:
            i = queue.pop()
            self.nodes[i]["swc_id"] = swc_id
            visited.add(i)
            for j in [j for j in self.neighbors(i) if j not in visited]:
                queue.append(j)

    def n_nearby_leafs(self, proposal, radius):
        """
        Counts the number of nearby leaf nodes within a specified radius from
        a proposal.

        Parameters
        ----------
        proposal : frozenset
            Pproposal for which nearby leaf nodes are to be counted.
        radius : float
            The radius within which to search for nearby leaf nodes.

        Returns
        -------
        int
            Number of nearby leaf nodes within a specified radius from
            a proposal.

        """
        xyz = self.proposal_midpoint(proposal)
        return len(self.query_kdtree(xyz, radius, "leaf")) - 1

    # --- util ---
    def is_soma(self, node_or_swc):
        """
        Determines whether "node_or_swc" corresponds to a soma.

        Parameters
        ----------
        node_or_swc : str
            node or swc id to be checked.

        Returns
        -------
        bool
            Indication of whether "node_or_swc" corresponds to a soma.

        """
        assert type(node_or_swc) in [int, str]
        if type(node_or_swc) is int:
            swc_id = self.nodes[node_or_swc]["swc_id"]
            return swc_id in self.soma_ids.keys()
        else:
            return node_or_swc in self.soma_ids.keys()

    def dist(self, i, j):
        """
        Computes the Euclidean distance between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node in self.
        j : int
            Nonde in self.

        Returns
        -------
        float
            Euclidean distance between nodes "i" and "j".

        """
        return get_dist(self.nodes[i]["xyz"], self.nodes[j]["xyz"])

    def branches(self, i, ignore_reducibles=True, key="xyz"):
        branches = list()
        for j in self.neighbors(i):
            branch = self.oriented_edge((i, j), i, key=key)
            if ignore_reducibles:
                root = i
                while self.degree[j] == 2:
                    k = self.get_other_nb(j, root)
                    branch_jk = self.oriented_edge((j, k), j, key=key)
                    if key == "xyz":
                        branch = np.vstack([branch, branch_jk])
                    else:
                        branch = np.concatenate((branch, branch_jk))
                    root = j
                    j = k
            branches.append(branch)
        return branches

    def branch(self, leaf, key="xyz"):
        """
        Gets the xyz coordinates or radii contained in the edge emanating from
        "leaf".

        Parameters
        ----------
        leaf : int
            Leaf node.

        Returns
        ----------
        numpy.ndarray
            xyz coordinates or radii contained in the edge emanating from
            "leaf".

        """
        assert self.is_leaf(leaf)
        return self.branches(leaf, key=key)[0]

    def get_other_nb(self, i, j):
        """
        Gets the other neighbor of node "i" which is not "j" such that "j" is
        a neighbor of node "i".

        Parameters
        ----------
        i : int
            Node with degree 2.
        j : int
            Neighbor of node "i"

        Returns
        -------
        int
            Neighbor of node "i" which is not "j".

        """
        assert self.degree[i] == 2, "Node does not have degree 2."
        nbs = list(self.neighbors(i))
        nbs.remove(j)
        return nbs[0]

    def oriented_edge(self, edge, i, key="xyz"):
        if (self.edges[edge][key][0] == self.nodes[i][key]).all():
            return self.edges[edge][key]
        else:
            return np.flip(self.edges[edge][key], axis=0)

    def is_contained(self, node_or_xyz, buffer=0):
        if self.bbox:
            coord = self.to_voxels(node_or_xyz)
            return util.is_contained(self.bbox, coord, buffer=buffer)
        else:
            return True

    def branch_contained(self, xyz_list):
        if self.bbox:
            return all(
                [self.is_contained(xyz, buffer=-32) for xyz in xyz_list]
            )
        else:
            return True

    def to_voxels(self, node_or_xyz, shift=False):
        shift = self.origin if shift else np.zeros((3))
        if type(node_or_xyz) is int:
            coord = img_util.to_voxels(self.nodes[node_or_xyz]["xyz"])
        else:
            coord = img_util.to_voxels(node_or_xyz)
        return coord - shift

    def is_leaf(self, i):
        """
        Checks whether node "i" is a leaf.

        Parameters
        ----------
        i : int
            Node to be checked.

        Returns
        -------
        Indication of whether node "i" is a leaf

        """
        return True if self.degree[i] == 1 else False

    def leaf_neighbor(self, i):
        """
        Gets the unique neighbor of the leaf node "i".

        Parameters
        ----------
        i : int
            Leaf  node.

        Returns
        -------
        int
             Unique neighbor of the leaf node "i".

        """
        assert self.is_leaf(i)
        return list(self.neighbors(i))[0]

    def xyz_to_swc(self, xyz, return_node=False):
        if tuple(xyz) in self.xyz_to_edge.keys():
            edge = self.xyz_to_edge[tuple(xyz)]
            i, j = tuple(edge)
            if return_node:
                return self.nodes[i]["swc_id"], i
            else:
                return self.nodes[i]["swc_id"]
        else:
            return None

    def component_path_length(self, root):
        path_length = 0
        for i, j in nx.dfs_edges(self, source=root):
            path_length += self.edges[i, j]["length"]
        return path_length

    # --- write graph to swcs ---
    def to_zipped_swcs(self, zip_path, color=None, min_size=0):
        with zipfile.ZipFile(zip_path, "w") as zip_writer:
            cnt = 0
            for nodes in nx.connected_components(self):
                root = util.sample_once(nodes)
                if self.component_path_length(root) > min_size:
                    self.to_zipped_swc(zip_writer, nodes, color)
                    cnt += 1
            return cnt

    def to_zipped_swc(self, zip_writer, nodes, color):
        with StringIO() as text_buffer:
            # Preamble
            n_entries = 0
            node_to_idx = dict()
            if color:
                text_buffer.write("# COLOR " + color)
            text_buffer.write("# id, type, z, y, x, r, pid")

            # Write entries
            for i, j in nx.dfs_edges(self.subgraph(nodes)):
                # Root entry
                if n_entries == 0:
                    swc_id = self.nodes[i]["swc_id"]
                    x, y, z = tuple(self.nodes[i]["xyz"])
                    r = 6 if self.nodes[i]["radius"] == 7.3141592 else 2

                    text_buffer.write("\n" + f"1 2 {x} {y} {z} {r} -1")
                    node_to_idx[i] = 1
                    n_entries += 1

                # Entry
                parent = node_to_idx[i]
                text_buffer, n_entries = self.branch_to_zip(
                    text_buffer, n_entries, i, j, parent, color
                )
                node_to_idx[j] = n_entries
            zip_writer.writestr(f"{swc_id}.swc", text_buffer.getvalue())

    def branch_to_zip(self, text_buffer, n_entries, i, j, parent, color):
        # Orient branch
        branch_xyz = self.edges[i, j]["xyz"]
        branch_radius = self.edges[i, j]["radius"]
        if (branch_xyz[0] != self.nodes[i]["xyz"]).any():
            branch_xyz = np.flip(branch_xyz, axis=0)
            branch_radius = np.flip(branch_radius, axis=0)

        # Make entries
        idxs = np.arange(1, len(branch_xyz))
        for k in util.spaced_idxs(idxs, 6):
            x, y, z = tuple(branch_xyz[k])
            r = 6 if branch_radius[k] == 7.3141592 else 2

            node_id = n_entries + 1
            parent = n_entries if k > 1 else parent
            text_buffer.write("\n" + f"{node_id} 2 {x} {y} {z} {r} {parent}")
            n_entries += 1
        return text_buffer, n_entries

    def save_labels(self, path):
        with open(path, "w") as f:
            for swc_id in self.swc_ids:
                f.write(f"{swc_id}\n")


# -- util --
def avg_radius(radii_list):
    avg = 0
    for radii in radii_list:
        end = max(min(16, len(radii) - 1), 1)
        avg += np.mean(radii[0:end]) / len(radii_list)
    return avg


def directional_origin(branch_1, branch_2):
    origin_1 = np.mean(branch_1, axis=0)
    origin_2 = np.mean(branch_2, axis=0)
    return np.mean(np.vstack(origin_1, origin_2), axis=0)
