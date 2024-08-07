"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Implementation of subclass of Networkx.Graph called "NeuroGraph".

"""
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import StringIO

import networkx as nx
import numpy as np
import tensorstore as ts
from scipy.spatial import KDTree

from deep_neurographs import generate_proposals, geometry
from deep_neurographs import graph_utils as gutils
from deep_neurographs import img_utils, swc_utils, utils
from deep_neurographs.geometry import dist as get_dist
from deep_neurographs.geometry import get_midpoint
from deep_neurographs.machine_learning.groundtruth_generation import (
    init_targets,
)

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
        node_spacing=1,
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
        self.proposals = set()
        self.target_edges = set()
        self.node_cnt = 0
        self.node_spacing = node_spacing
        self.soma_ids = dict()

        # Initialize data structures for proposals
        self.xyz_to_edge = dict()
        self.kdtree = None
        self.leaf_kdtree = None
        self.merged_ids = set()

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
        for i in gutils.largest_components(self, k):
            self.soma_ids[self.nodes[i]["swc_id"]] = i

    # --- Edit Graph --
    def add_component(self, irreducibles):
        """
        Adds a connected component to "self".

        Parameters
        ----------
        irreducibles : dict
            Dictionary containing the irreducibles of some connected component
            being added to "self". This dictionary must contain the keys:
            'leafs', 'junctions', 'edges', and 'swc_id'.

        Returns
        -------
        None

        """
        swc_id = irreducibles["swc_id"]
        if swc_id not in self.swc_ids:
            # Nodes
            self.swc_ids.add(swc_id)
            ids = self.__add_nodes(irreducibles, "leafs", dict())
            ids = self.__add_nodes(irreducibles, "junctions", ids)

            # Edges
            for (i, j), attrs in irreducibles["edges"].items():
                edge = (ids[i], ids[j])
                idxs = np.arange(0, attrs["xyz"].shape[0], self.node_spacing)
                if idxs[-1] != attrs["xyz"].shape[0] - 1:
                    idxs = np.append(idxs, attrs["xyz"].shape[0] - 1)
                self.__add_edge(edge, attrs, idxs, swc_id)

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
            'leafs' or 'junctions'.
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
            if node_type == "leafs":
                self.leafs.add(cur_id)
            else:
                self.junctions.add(cur_id)
            node_ids[i] = cur_id
        return node_ids

    def __add_edge(self, edge, attrs, idxs, swc_id):
        """
        Adds an edge to "self".

        Parameters
        ----------
        edge : tuple
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" that were obtained from an swc
            file.
        idxs : dict
            Indices of attributes to store in order to reduce the amount of
            memory required to store "self".
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
            radius=attrs["radius"][idxs],
            xyz=attrs["xyz"][idxs],
            swc_id=swc_id,
        )
        for xyz in attrs["xyz"][idxs]:
            self.xyz_to_edge[tuple(xyz)] = edge

    """
    def absorb_node(self, i, nb_1, nb_2):
        # Get attributes
        xyz = self.get_branches(i, key="xyz")
        radius = self.get_branches(i, key="radius")

        # Edit graph
        self.remove_node(i)
        self.add_edge(
            nb_1,
            nb_2,
            xyz=np.vstack([np.flip(xyz[1], axis=0), xyz[0][1:, :]]),
            radius=np.concatenate((radius[0], np.flip(radius[1]))),
            swc_id=self.nodes[nb_1]["swc_id"],
        )
    """

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
        idxs_1 = np.arange(0, idx + 1)
        idxs_2 = np.arange(idx, len(attrs["xyz"]))
        self.__add_edge((i, node_id), attrs, idxs_1, swc_id)
        self.__add_edge((node_id, j), attrs, idxs_2, swc_id)
        return node_id

    # --- Proposal Generation ---
    def generate_proposals(
        self,
        search_radius,
        complex_bool=False,
        long_range_bool=False,
        proposals_per_leaf=3,
        optimize=False,
        optimization_depth=10,
        return_trimmed_proposals=False,
        trim_endpoints_bool=False,
    ):
        """
        Generates proposals from leaf nodes in "self".

        Parameters
        ----------
        search_radius : float
            Search radius used to generate proposals.
        complex_bool : bool, optional
            Indication of whether to generate complex proposals. The default
            is False.
        long_range_bool : bool, optional
            Indication of whether to generate long range proposals. The
            default is False.
        proposals_per_leaf : int, optional
            Maximum number of proposals generated for each leaf. The default
            is False.
        optimize : bool, optional
            Indication of whether to optimize proposal alignment to image. The
            default is False.
        optimization_depth : int, optional
            Depth to check during optimization. The default is False.
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
        generate_proposals.run(
            self,
            search_radius,
            complex_bool=complex_bool,
            long_range_bool=long_range_bool,
            trim_endpoints_bool=trim_endpoints_bool,
        )

        # Delete large data structures
        # Finish
        # absorb reducible nodes
        if optimize:
            self.run_optimization()

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

    def init_targets(self, target_neurograph):
        target_neurograph.init_kdtree()
        self.target_edges = init_targets(target_neurograph, self)

    def run_optimization(self):
        driver = "n5" if "n5" in self.img_path else "zarr"
        img = utils.get_superchunk(
            self.img_path, driver, self.origin, self.shape, from_center=False
        )
        for edge in self.proposals:
            xyz_1, xyz_2 = geometry.optimize_alignment(self, img, edge)
            self.proposals[edge]["xyz"] = np.array([xyz_1, xyz_2])

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
            xyz_list = [self.nodes[i]["xyz"] for i in self.leafs]
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

    def get_projection(self, xyz):
        """
        Gets the xyz coordinates of the nearest neighbor of "xyz".

        Parameters
        ----------
        xyz : tuple
            xyz coordinate to be queried.

        Returns
        -------
        tuple
            xyz coordinate of the nearest neighbor of "xyz".

        """
        _, idx = self.kdtree.query(xyz, k=1)
        return tuple(self.kdtree.data[idx])

    # --- Proposal Utils ---
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

    def get_simple_proposals(self):
        return set([e for e in self.proposals if self.is_simple(e)])

    def get_complex_proposals(self):
        return set([e for e in self.proposals if not self.is_simple(e)])

    def proposal_search(self, root_1, root_2, max_depth, max_dist):
        queue = [(root_1, 0), (root_2, 0)]
        roots = [root_1, root_2]
        proposals = set()
        visited = set()
        while len(queue) > 0:
            # Visit node
            i, depth = queue.pop(0)
            proposals = proposals.union(self.nodes[i]["proposals"])
            visited.add(i)

            # Add neighbors
            if depth < max_depth:
                for j in [j for j in self.neighbors(i) if j not in visited]:
                    dist = max(self.dist(root_1, j), self.dist(root_2, j))
                    if j not in roots and dist < max_dist:
                        queue.append((j, depth + 1))
        return len(proposals) - 2 > 0

    def is_simple(self, edge):
        i, j = tuple(edge)
        return True if self.is_leaf(i) and self.is_leaf(j) else False

    def proposal_length(self, proposal):
        return self.dist(*tuple(proposal))

    def proposal_midpoint(self, proposal):
        i, j = tuple(proposal)
        return get_midpoint(self.nodes[i]["xyz"], self.nodes[j]["xyz"])

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
        radii_i = self.get_branches(i, ignore_reducibles=True, key="radius")
        radii_j = self.get_branches(j, ignore_reducibles=True, key="radius")
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

    def proposal_directionals(self, proposal, window):
        # Compute tangent vectors
        i, j = tuple(proposal)
        direction = geometry.tangent(self.proposal_xyz(proposal))
        origin = self.proposal_midpoint(proposal)
        branches_i = self.get_branches(i, ignore_reducibles=True)
        branches_j = self.get_branches(j, ignore_reducibles=True)
        direction_i = geometry.get_directional(branches_i, i, origin, window)
        direction_j = geometry.get_directional(branches_j, j, origin, window)

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

    def merge_proposal(self, edge):
        i, j = tuple(edge)
        swc_id_i = self.nodes[i]["swc_id"]
        swc_id_j = self.nodes[j]["swc_id"]
        if not (self.is_soma(i) and self.is_soma(j)):
            # Attributes
            xyz = np.vstack([self.nodes[i]["xyz"], self.nodes[j]["xyz"]])
            radius = np.array(
                [self.nodes[i]["radius"], self.nodes[j]["radius"]]
            )
            swc_id = swc_id_i if self.is_soma(i) else swc_id_j

            # Update graph
            self.merged_ids.add((swc_id_i, swc_id_j))
            self.upd_ids(swc_id, j if swc_id == swc_id_i else i)
            self.add_edge(i, j, xyz=xyz, radius=radius, swc_id=swc_id)
            if i in self.leafs:
                self.leafs.remove(i)
            if j in self.leafs:
                self.leafs.remove(j)
            self.proposals.remove(edge)

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
        xyz = self.proposal_midpoint(proposal)
        return len(self.query_kdtree(xyz, radius, "leaf")) - 1

    # --- Utils ---
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

    def get_branches(self, i, ignore_reducibles=False, key="xyz"):
        branches = []
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

    def get_branch(self, leaf, key="xyz"):
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
        return self.get_branches(leaf, key=key)[0]

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
    """
    def node_xyz_dist(self, node, xyz):
        return get_dist(xyz, self.nodes[node]["xyz"])
    """

    def edge_length(self, edge):
        """
        Computes length of path stored as xyz coordinates in "edge".

        Parameters
        ----------
        edge : tuple
            Edge in self.

        Returns
        -------
        float
            Path length of edge.

        """
        return geometry.path_length(self.edges[edge]["xyz"])

    def is_contained(self, node_or_xyz, buffer=0):
        if self.bbox:
            coord = self.to_voxels(node_or_xyz)
            return utils.is_contained(self.bbox, coord, buffer=buffer)
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
        if type(node_or_xyz) == int:
            coord = img_utils.to_voxels(self.nodes[node_or_xyz]["xyz"])
        else:
            coord = img_utils.to_voxels(node_or_xyz)
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

    def get_edge_attr(self, edge, key):
        xyz_arr = gutils.get_edge_attr(self, edge, key)
        return xyz_arr[0], xyz_arr[-1]

    def to_patch_coords(self, edge, midpoint, chunk_size):
        patch_coords = []
        for xyz in self.edges[edge]["xyz"]:
            coord = self.to_voxels(xyz)
            local_coord = utils.voxels_to_patch(coord, midpoint, chunk_size)
            patch_coords.append(local_coord)
        return patch_coords

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

    def component_cardinality(self, root):
        cardinality = 0
        queue = [(-1, root)]
        visited = set()
        while len(queue):
            # Visit
            i, j = queue.pop()
            visited.add(frozenset((i, j)))
            if i != -1:
                cardinality = len(self.edges[i, j]["xyz"])

            # Add neighbors
            for k in self.neighbors(j):
                if frozenset((j, k)) not in visited:
                    queue.append((j, k))
        return cardinality

    # --- write graph to swcs ---
    def to_zipped_swcs(self, zip_path, color=None):
        n_components = utils.reformat_number(gutils.count_components(self))
        print(f"Writing {n_components} swcs to local machine!")
        with zipfile.ZipFile(zip_path, "w") as zip_writer:
            for nodes in nx.connected_components(self):
                self.to_zipped_swc(zip_writer, nodes, color)

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
                    r = self.nodes[i]["radius"]
                    if color != "1.0 0.0 0.0":
                        r += 1.5
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

    def to_swcs(self, swc_dir):
        """
        Write a neurograph to "swc_dir" such that each connected component is
        saved as an swc file.

        Parameters
        ----------
        swc_dir : str
            Directory that neurograph is to be written to

        Returns
        -------
        None

        """
        with ThreadPoolExecutor() as executor:
            threads = []
            n_components = gutils.count_components(self)
            print(f"Writing {n_components} swcs to local machine!")
            for i, nodes in enumerate(nx.connected_components(self)):
                threads.append(executor.submit(self.to_swc, swc_dir, nodes))

    def to_swc(self, swc_dir, nodes, color=None):
        """
        Generates list of swc entries for a given connected component.

        Parameters
        ----------
        swc_dir : str
            Directory that swc will be written to.
        nodes : list[int]
            Nodes to be written to an swc file.
        color : None or str
            Color that swc files should be given.

        Returns
        -------
        None.

        """
        entry_list = []
        node_to_idx = dict()
        for i, j in nx.dfs_edges(self.subgraph(nodes)):
            # Initialize
            if len(entry_list) == 0:
                x, y, z = tuple(self.nodes[i]["xyz"])
                r = self.nodes[i]["radius"]
                entry_list.append(f"1 2 {x} {y} {z} {r} -1")
                node_to_idx[i] = 1

                filename = self.nodes[i]["swc_id"] + ".swc"
                path = os.path.join(swc_dir, filename)

            # Create entry
            parent = node_to_idx[i]
            entry_list = self.branch_to_entries(entry_list, i, j, parent)
            node_to_idx[j] = len(entry_list)

        # Write
        swc_utils.write(path, entry_list, color=color)

    def branch_to_entries(self, entry_list, i, j, parent):
        # Orient branch
        branch_xyz = self.edges[i, j]["xyz"]
        branch_radius = self.edges[i, j]["radius"]
        if (branch_xyz[0] != self.nodes[i]["xyz"]).any():
            branch_xyz = np.flip(branch_xyz, axis=0)
            branch_radius = np.flip(branch_radius, axis=0)

        # Make entries
        for k in range(1, len(branch_xyz)):
            x, y, z = tuple(branch_xyz[k])
            r = branch_radius[k]
            node_id = len(entry_list) + 1
            parent = len(entry_list) if k > 1 else parent
            entry = f"{node_id} 2 {x} {y} {z} {r} {parent}"
            entry_list.append(entry)
        return entry_list

    def branch_to_zip(self, text_buffer, n_entries, i, j, parent, color):
        # Orient branch
        branch_xyz = self.edges[i, j]["xyz"]
        branch_radius = self.edges[i, j]["radius"]
        if (branch_xyz[0] != self.nodes[i]["xyz"]).any():
            branch_xyz = np.flip(branch_xyz, axis=0)
            branch_radius = np.flip(branch_radius, axis=0)

        # Make entries
        for k in range(1, len(branch_xyz)):
            x, y, z = tuple(branch_xyz[k])
            r = branch_radius[k]
            if color != "1.0 0.0 0.0":
                r += 1
            node_id = n_entries + 1
            parent = n_entries if k > 1 else parent
            text_buffer.write("\n" + f"{node_id} 2 {x} {y} {z} {r} {parent}")
            n_entries += 1
        return text_buffer, n_entries


# -- utils --
def avg_radius(radii_list):
    avg = 0
    for radii in radii_list:
        end = max(min(16, len(radii) - 1), 1)
        avg += np.mean(radii[0:end]) / len(radii_list)
    return avg
