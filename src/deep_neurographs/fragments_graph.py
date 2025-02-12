"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of a custom subclass of Networkx.Graph called "FragmentsGraph".
After initializing an instance of this subclass, the graph is built by reading
and processing SWC files (i.e. neuron fragments). It then stores the relevant
information into the graph structure.

    Graph Construction Algorithm:
        1. Read Neuron Fragments
            Reads SWC files and stores the contents in a dictionary with the
            keys: "id", "xyz", "radius", "pid", and "swc_id". Each SWC file is
            assumed to contain uniformly spaced points, each separated by 1
            voxel.

        2. Extract Irreducibles
            Finds the components of the irreducible subgraph from each SWC
            file. The irreducible components of a graph are the following:
                (1) Leafs: Nodes of degree 1
                (2) Branchings: Nodes of degree 3+
                (3) Edges: Paths between irreducible nodes

        3. Add Irreducibles
            to do...

"""

from copy import deepcopy
from io import StringIO
from numpy import concatenate
from scipy.spatial import KDTree
from tqdm import tqdm

import networkx as nx
import numpy as np
import zipfile

from deep_neurographs import proposal_generation
from deep_neurographs.utils import (
    geometry_util as geometry,
    graph_util as gutil,
    swc_util,
    util,
)
from deep_neurographs.machine_learning import groundtruth_generation


class FragmentsGraph(nx.Graph):
    """
    Custom subclass of NetworkX.Graph constructed from neuron fragments. The
    graph's nodes are irreducible, meaning each node has either degree 1
    (leaf) or degree 3+ (branching points). Each edge has an attribute that
    stores a dense path of points connecting irreducible nodes. Additionally,
    the graph has an attribute called "proposals", which is a set of potential
    connections between pairs of neuron fragments.

    """

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_size=30.0,
        node_spacing=1,
        prune_depth=16.0,
        remove_high_risk_merges=False,
        segmentation_path=None,
        smooth_bool=True,
        somas_path=None,
        verbose=False,
    ):
        """
        Initializes an instance of FragmentsGraph.

        Parameters
        ----------
        anisotropy : Tuple[int], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is (1.0, 1.0, 1.0).
        min_size : float, optional
            Minimum path length of swc files that are loaded into the
            FragmentsGraph. The default is 30.0 microns.
        node_spacing : int, optional
            Sampling rate for nodes in FragmentsGraph. Every "node_spacing"
            node is retained.
        prune_depth : int, optional
            Branches with length less than "prune_depth" microns are removed.
            The default is 16.0 microns.
        remove_high_risk_merges : bool, optional
            Indication of whether to remove high risk merge sites (i.e. close
            branching points). The default is False.
        segmentation_path : str, optional
            Path to segmentation stored in GCS bucket. The default is None.
        smooth_bool : bool, optional
            Indication of whether to smooth xyz coordinates from SWC files.
            The default is True.
        somas_path : str, optional
            Path to a txt file containing xyz coordinates of detected somas.
            The default is None.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            FragmentsGraph. The default is True.

        Returns
        -------
        None

        """
        # Call parent class
        super(FragmentsGraph, self).__init__()

        # Loaders
        self.graph_loader = gutil.GraphLoader(
            anisotropy=anisotropy,
            min_size=min_size,
            node_spacing=node_spacing,
            prune_depth=prune_depth,
            remove_high_risk_merges=remove_high_risk_merges,
            segmentation_path=segmentation_path,
            smooth_bool=smooth_bool,
            somas_path=somas_path,
            verbose=verbose,
        )
        self.swc_reader = swc_util.Reader(anisotropy, min_size)

        # Instance attributes - Graph
        self.anisotropy = anisotropy
        self.leaf_kdtree = None
        self.node_cnt = 0
        self.node_spacing = node_spacing
        self.soma_ids = set()
        self.swc_ids = set()
        self.verbose = verbose
        self.xyz_to_edge = dict()

        # Instance attributes - Proposals
        self.merged_ids = set()
        self.proposals = set()

    # --- Build Graph --
    def load_fragments(self, fragments_pointer):
        """
        Loads fragments into "self" by reading SWC files stored on either the
        cloud or local machine, then extracts the irreducible components from
        from each SWC file.

        Parameters
        ----------
        fragments_pointer : Any
            Pointer to SWC files to be loaded, see "swc_util.Reader" for
            documentation.

        Returns
        -------
        None

        """
        swc_dicts = self.swc_reader.load(fragments_pointer)
        irreducibles_list = self.graph_loader.extract_irreducibles(swc_dicts)
        while len(irreducibles_list) > 0:
            irreducibles = irreducibles_list.pop()
            self.add_irreducibles(irreducibles)

    def add_irreducibles(self, irreducibles):
        """
        Adds the irreducibles from a single connected component to "self".

        Parameters
        ----------
        irreducibles : dict
            Dictionary containing the irreducibles of a connected component to
            be added to "self". This dictionary must contain the keys: "leaf",
            "branching", "edge", "swc_id", and "is_soma".

        Returns
        -------
        None

        """
        # Nodes
        ids = self.__add_nodes(irreducibles, "leaf", dict())
        ids = self.__add_nodes(irreducibles, "branching", ids)

        # Edges
        swc_id = irreducibles["swc_id"]
        for (i, j), attrs in irreducibles["edge"].items():
            edge = (ids[i], ids[j])
            self.__add_edge(edge, attrs, swc_id)

        # SWC ID
        self.swc_ids.add(swc_id)
        if irreducibles["is_soma"]:
            self.soma_ids.add(swc_id)

    # --- update graph structure ---
    def __add_nodes(self, irreducibles, node_type, node_ids):
        """
        Adds a set of nodes from "irreducibles" to "self".

        Parameters
        ----------
        irreducibles : dict
            Dictionary containing the irreducibles of some connected component
            being added to "self".
        node_type : str
            Type of node being added to "self". Note: value must be either
            'leaf' or 'branching'.
        node_ids : dict
            Dictionary containing conversion from a node id in "irreducibles"
            to the corresponding node id in "self".

        Returns
        -------
        dict
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
        edge : Tuple[int]
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an swc file.
        swc_id : str
            SWC ID corresponding to edge.

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
        None

        """
        # NOT USED AND OUTDATED
        nodes = deepcopy(self.nodes)
        for i in tqdm(nodes):
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
        int
            Node ID of node that was created.

        """
        # Remove old edge
        (i, j) = edge
        self.remove_edge(i, j)

        # Create node
        node_id = self.node_cnt + 1
        self.add_node(
            node_id,
            proposals=set(),
            radius=attrs["radius"][idx],
            swc_id=attrs["swc_id"],
            xyz=tuple(attrs["xyz"][idx]),
        )
        self.node_cnt += 1

        # Create edges
        n = len(attrs["xyz"])
        attrs_1 = {k: v[np.arange(idx + 1)] for k, v in attrs.items()}
        attrs_2 = {k: v[np.arange(idx, n)] for k, v in attrs.items()}
        self.__add_edge((i, node_id), attrs_1, attrs["swc_id"])
        self.__add_edge((node_id, j), attrs_2, attrs["swc_id"])
        return node_id

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

    # --- Proposal Generation ---
    def generate_proposals(
        self,
        search_radius,
        complex_bool=False,
        groundtruth_graph=None,
        long_range_bool=False,
        proposals_per_leaf=3,
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
        proposals_per_leaf : int, optional
            Maximum number of proposals generated for each leaf. The default
            is 3.
        trim_endpoints_bool : bool, optional
            Indication of whether to trim endpoints. The default is False.

        Returns
        -------
        None

        """
        # Main
        self.reset_proposals()
        self.set_proposals_per_leaf(proposals_per_leaf)
        proposal_generation.run(
            self,
            search_radius,
            complex_bool=complex_bool,
            long_range_bool=long_range_bool,
            trim_endpoints_bool=trim_endpoints_bool,
        )

        # Set groundtruth
        if groundtruth_graph:
            self.gt_accepts = groundtruth_generation.run(
                self, groundtruth_graph
            )
        else:
            self.gt_accepts = set()

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
            Node ID.
        j : int
            Node ID

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
        proposal : Frozenset[int]
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
        proposal : Frozenset[int]
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
            Leaf node ID.
        i : int
            NOde ID.
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

    def proposal_connected_component(self, proposal):
        """
        Extracts the connected component that "proposal" belongs to in the
        proposal induced subgraph.

        Parameters
        ----------
        proposal : frozenset
            Proposal used to as the root to extract its connected component
            in the proposal induced subgraph.

        Returns
        -------
        List[Frozenset[int]]
            List of proposals in the connected component that "proposal"
            belongs to in the proposal induced subgraph.

        """
        queue = [proposal]
        visited = set()
        while len(queue) > 0:
            # Visit proposal
            p = queue.pop()
            visited.add(p)

            # Update queue
            for i in p:
                for j in self.nodes[i]["proposals"]:
                    p_ij = frozenset({i, j})
                    if p_ij not in visited:
                        queue.append(p_ij)
        return visited

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
        """
        Determines whether both nodes in a proposal are leafs.

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of nodes that form a proposal.

        Returns
        -------
        bool
            Indication of whether both nodes in a proposal are leafs.

        """
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

    def proposal_attr(self, proposal, attr):
        """
        Gets the attributes of nodes in "proposal".

        Parameters
        ----------
        proposal : Frozenset[int]
            Pair of nodes that form a proposal.
        attr : str
            Name of attribute to be returned.

        Returns
        -------
        numpy.ndarray
            Attributes of nodes in "proposal".

        """
        i, j = tuple(proposal)
        if attr == "swc_id":
            attrs = [
                reformat(self.nodes[i][attr]), reformat(self.nodes[j][attr])
            ]
        else:
            attrs = np.array([self.nodes[i][attr], self.nodes[j][attr]])
        return attrs

    def proposal_avg_radii(self, proposal):
        i, j = tuple(proposal)
        radii_i = self.branches(i, key="radius")
        radii_j = self.branches(j, key="radius")
        return np.array([avg_radius(radii_i), avg_radius(radii_j)])

    def proposal_directionals(self, proposal, depth):
        # Extract points along branches
        i, j = tuple(proposal)
        xyz_list_i = [geometry.truncate_path(b, depth) for b in self.branches(i)]
        xyz_list_j = [geometry.truncate_path(b, depth) for b in self.branches(j)]
        origin = self.proposal_midpoint(proposal)

        # Compute tangent vectors
        direction_i = geometry.get_directional(xyz_list_i, origin, depth)
        direction_j = geometry.get_directional(xyz_list_j, origin, depth)
        direction = geometry.tangent(self.proposal_attr(proposal, "xyz"))

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
            self.nodes[i]["radius"] = 5.3141592
            self.nodes[j]["radius"] = 5.3141592
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
            Node ID

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

    # --- miscellaneous ---
    def dist(self, i, j):
        """
        Computes the Euclidean distance between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Nonde ID.

        Returns
        -------
        float
            Euclidean distance between nodes "i" and "j".

        """
        return geometry.dist(self.nodes[i]["xyz"], self.nodes[j]["xyz"])

    def find_fragments_by_ids(self, swc_ids):
        fragments = dict()
        for nodes in nx.connected_components(self):
            i = util.sample_once(nodes)
            if self.nodes[i]["swc_id"] in swc_ids:
                swc_id = self.nodes[i]["swc_id"]
                fragments[swc_id] = nodes
        return fragments

    def get_leafs(self):
        """
        Gets all leaf nodes in graph.

        Parameters
        ----------
        None

        Returns
        -------
        List[int]
            Leaf nodes in graph.

        """
        return [i for i in self.nodes if self.is_leaf(i)]

    def is_leaf(self, i):
        """
        Checks if node "i" is a leaf.

        Parameters
        ----------
        i : int
            Node to be checked.

        Returns
        -------
        bool
            Indication of whether node "i" is a leaf

        """
        return True if self.degree[i] == 1 else False

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
        assert type(node_or_swc) in [int, str], "Type error!"
        if isinstance(node_or_swc, int):
            node_or_swc = self.nodes[node_or_swc]["swc_id"]
        return node_or_swc in self.soma_ids

    def path_length(self, root):
        """
        Computes the path length of the connected component containing "root".

        Parameters
        ----------
        root : int
            Node in connected component to be search.

        Returns
        -------
        float
            Path length of connected component containing "root".

        """
        path_length = 0
        for i, j in nx.dfs_edges(self, source=root):
            path_length += self.edges[i, j]["length"]
        return path_length

    def branches(self, i, key="xyz"):
        """
        rename connected_edge_attr
        """
        attrs = list()
        for j in self.neighbors(i):
            attr_ij = self.oriented_edge_attr((i, j), i, key=key)
            root = i
            while self.degree[j] == 2:
                k = self.get_other_nb(j, root)
                attr_jk = self.oriented_edge_attr((j, k), j, key=key)
                if key == "xyz":
                    attr_ij = np.vstack([attr_ij, attr_jk])
                else:
                    attr_ij = np.concatenate((attr_ij, attr_jk))
                root = j
                j = k
            attrs.append(attr_ij)
        return attrs

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
        assert self.degree[i] == 2, "deg(i) != 2."
        nbs = list(self.neighbors(i))
        nbs.remove(j)
        return nbs[0]

    def oriented_edge_attr(self, edge, i, key="xyz"):
        assert i in edge
        if (self.edges[edge][key][0] == self.nodes[i][key]).all():
            return self.edges[edge][key]
        else:
            return np.flip(self.edges[edge][key], axis=0)

    def leaf_neighbor(self, i):
        """
        Gets the unique neighbor of the leaf node "i".

        Parameters
        ----------
        i : int
            Leaf node.

        Returns
        -------
        int
             Unique neighbor of the leaf node "i".

        """
        assert self.is_leaf(i)
        return list(self.neighbors(i))[0]

    def xyz_to_id(self, xyz, return_node=False):
        if tuple(xyz) in self.xyz_to_edge.keys():
            edge = self.xyz_to_edge[tuple(xyz)]
            i, j = tuple(edge)
            if return_node:
                return self.nodes[i]["swc_id"], i
            else:
                return self.nodes[i]["swc_id"]
        else:
            return None

    # --- write graph to swcs ---
    def to_zipped_swcs(
        self,
        zip_path,
        color=None,
        min_size=0,
        preserve_radius=False,
        sampling_rate=1,
    ):
        with zipfile.ZipFile(zip_path, "w") as zip_writer:
            cnt = 0
            for nodes in nx.connected_components(self):
                root = util.sample_once(nodes)
                if self.path_length(root) > min_size:
                    self.to_zipped_swc(
                        zip_writer,
                        nodes,
                        color=color,
                        preserve_radius=preserve_radius,
                        sampling_rate=sampling_rate
                    )
                    cnt += 1
            return cnt

    def to_zipped_swc(
        self,
        zip_writer,
        nodes,
        color=None,
        prefix="",
        preserve_radius=False,
        sampling_rate=1,
    ):
        with StringIO() as text_buffer:
            # Preamble
            n_entries = 0
            node_to_idx = dict()
            if color:
                text_buffer.write("# COLOR " + color)
            text_buffer.write("# id, type, x, y, z, r, pid")

            # Write entries
            for i, j in nx.dfs_edges(self.subgraph(nodes)):
                # Root entry
                if n_entries == 0:
                    # Get attributes
                    x, y, z = tuple(self.nodes[i]["xyz"])
                    if preserve_radius:
                        r = self.nodes[i]["radius"]
                    else:
                        r = 5 if self.nodes[i]["radius"] == 5.3141592 else 2

                    # Write entry
                    text_buffer.write("\n" + f"1 2 {x} {y} {z} {r} -1")
                    node_to_idx[i] = 1
                    n_entries += 1

                # Remaining entries
                parent = node_to_idx[i]
                text_buffer, n_entries = self.branch_to_zip(
                    text_buffer,
                    n_entries,
                    i,
                    j,
                    parent,
                    color,
                    preserve_radius=preserve_radius,
                    sampling_rate=sampling_rate
                )
                node_to_idx[j] = n_entries

            # Write SWC file
            filename = prefix + self.nodes[i]["swc_id"]
            filename = util.set_zip_path(zip_writer, filename, ".swc")
            zip_writer.writestr(filename, text_buffer.getvalue())

    def branch_to_zip(
        self,
        text_buffer,
        n_entries,
        i,
        j,
        parent,
        color,
        preserve_radius=False,
        sampling_rate=1,
    ):
        branch_xyz = self.oriented_edge_attr((i, j), i, "xyz")
        branch_radius = self.oriented_edge_attr((i, j), i, "radius")
        for k in util.spaced_idxs(len(branch_xyz), sampling_rate):
            # Get attributes
            node_id = n_entries + 1
            parent = n_entries if k > 1 else parent
            x, y, z = tuple(branch_xyz[k])
            if preserve_radius:
                r = branch_radius[k]
            else:
                r = 5 if branch_radius[k] == 5.3141592 else 2

            # Write entry
            text_buffer.write("\n" + f"{node_id} 2 {x} {y} {z} {r} {parent}")
            n_entries += 1
        return text_buffer, n_entries

    def save_labels(self, path):
        with open(path, "w") as f:
            for swc_id in self.swc_ids:
                f.write(f"{swc_id}\n")


# -- Helpers --
def avg_radius(radii_list):
    avg = 0
    for radii in radii_list:
        end = max(min(16, len(radii) - 1), 1)
        avg += np.mean(radii[0:end]) / len(radii_list)
    return avg


def reformat(segment_id):
    return int(segment_id.split(".")[0])
