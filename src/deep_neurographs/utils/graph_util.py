"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Overview
--------
Code that loads and preprocesses neuron fragments stored as swc files, then
constructs a custom graph object called a "FragmentsGraph" from the fragments.

    Graph Construction Algorithm:
        1. Load Neuron Fragments
            to do...

        2. Extract Irreducibles
            to do...

        3. Build FragmentsGraph
            to do...

"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import sample

import networkx as nx
import numpy as np
from tqdm import tqdm

from deep_neurographs import geometry
from deep_neurographs.utils import img_util, swc_util, util


class GraphLoader:
    """
    Class that is used to build an instance of FragmentsGraph.

    """

    def __init__(
        self,
        anisotropy=[1.0, 1.0, 1.0],
        min_size=30.0,
        node_spacing=1,
        progress_bar=False,
        prune_depth=20.0,
        smooth_bool=True,
    ):
        """
        Builds a FragmentsGraph by reading swc files stored either on the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        min_size : float, optional
            Minimum path length of swc files which are stored as connected
            components in the FragmentsGraph. The default is 30.0 (microns).
        node_spacing : int, optional
            Spacing (in microns) between nodes. The default is 1.
        progress_bar : bool, optional
            Indication of whether to print out a progress bar while building
            graph. The default is True.
        prune_depth : int, optional
            Branches less than "prune_depth" microns are pruned if "prune" is
            True. The default is 20.0 (microns).
        smooth_bool : bool, optional
            Indication of whether to smooth branches from swc files. The
            default is True.

        Returns
        -------
        None

        """
        self.anisotropy = anisotropy
        self.min_size = min_size
        self.node_spacing = node_spacing
        self.progress_bar = progress_bar
        self.prune_depth = prune_depth
        self.smooth_bool = smooth_bool

        self.reader = swc_util.Reader(anisotropy, min_size)

    def run(self, fragments_pointer):
        """
        Builds a FragmentsGraph by reading swc files stored either on the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        fragments_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentsGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        FragmentsGraph
            FragmentsGraph generated from swc files.

        """
        from deep_neurographs.fragments_graph import FragmentsGraph

        # Load fragments and extract irreducibles
        swc_dicts = self.reader.load(fragments_pointer)
        irreducibles = self.get_irreducibles(swc_dicts)

        # Build FragmentsGraph
        fragments_graph = FragmentsGraph(node_spacing=self.node_spacing)
        while len(irreducibles):
            irreducible_set = irreducibles.pop()
            fragments_graph.add_component(irreducible_set)
        return fragments_graph

    # --- Graph structure extraction ---
    def get_irreducibles(self, swc_dicts):
        """
        Processes a list of swc dictionaries in parallel and extracts the
        parts of the irreducible subgraphs from each swc file.

        Parameters
        ----------
        swc_dicts : List[dict]
            List of dictionaries such that each contains the contents of an
            swc file.

        Returns
        -------
        List[dict]
            List of dictionaries such that each contains the components of an
            irreducible subgraph extracted from an swc file.

        """
        # Initializations
        if self.progress_bar:
            pbar = tqdm(total=len(swc_dicts), desc="Extract Graphs")

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign Processes
            i = 0
            processes = [None] * len(swc_dicts)
            while swc_dicts:
                swc_dict = swc_dicts.pop()
                processes[i] = executor.submit(
                    self.extract_irreducibles, swc_dict)
                i += 1

            # Store results
            irreducibles = list()
            for process in as_completed(processes):
                result = process.result()
                if result is not None:
                    irreducibles.append(result)
                if self.progress_bar:
                    pbar.update(1)
        return irreducibles

    def extract_irreducibles(self, swc_dict):
        """
        Gets the irreducible subgraph from the given swc file.

        Parameters
        ----------
        swc_dict : dict
            Contents of an swc file.

        Returns
        -------
        dict
            Dictionary that each contains the components of an irreducible
            subgraph.

        """
        graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)
        self.prune_branches(graph)
        if compute_path_length(graph) > self.min_size:
            # Irreducible nodes
            leafs, branchings = get_irreducible_nodes(graph)

            # Irreducible nodes
            edges = dict()
            root = None
            for (i, j) in nx.dfs_edges(graph, source=util.sample_once(leafs)):
                # Check for start of irreducible edge
                if root is None:
                    root = i
                    attrs = init_edge_attrs(graph, root)

                # Check for end of irreducible edge
                upd_edge_attrs(graph, attrs, i, j)
                if j in leafs or j in branchings:
                    # Smooth (if applicable)
                    attrs = to_numpy(attrs)
                    if self.smooth_bool:
                        smooth_branch(graph, attrs, root, j)

                    # Finish
                    edges[(root, j)] = attrs
                    root = None

            # Output
            irreducibles = {
                "leaf": set_node_attrs(graph, leafs),
                "branching": set_node_attrs(graph, branchings),
                "edge": set_edge_attrs(graph, edges),
                "swc_id": swc_dict["swc_id"],
            }
            return irreducibles
        else:
            return None

    def prune_branches(self, graph):
        """
        Prunes all short branches from "graph". A short branch is a path
        between a leaf and branching node where the path length is less than
        "self.prune_depth" microns.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be pruned.

        Returns
        -------
        networkx.Graph
            Graph with short branches pruned.

        """
        deleted_nodes = list()
        n_passes = 0
        while len(deleted_nodes) > 0 or n_passes < 2:
            # Visit leafs
            n_passes += 1
            deleted_nodes = list()
            for leaf in get_leafs(graph):
                branch = [leaf]
                length = 0
                for (i, j) in nx.dfs_edges(graph, source=leaf):
                    # Visit edge
                    length += compute_dist(graph, i, j)
                    if graph.degree(j) == 2:
                        branch.append(j)
                    elif graph.degree(j) > 2:
                        deleted_nodes.extend(branch)
                        graph.remove_nodes_from(branch)
                        break

                    # Check whether to stop
                    if length > self.prune_depth:
                        if n_passes == 1:
                            k = min(3, len(branch))
                            graph.remove_nodes_from(branch[0:k])
                        break


# --- Utils ---
def get_irreducible_nodes(graph):
    """
    Gets irreducible nodes (i.e. leafs and branchings) of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    set, set
        Nodes with degreee 1 and degree > 2.

    """
    leafs = set()
    branchings = set()
    for i in graph.nodes:
        if graph.degree[i] == 1:
            leafs.add(i)
        elif graph.degree[i] > 2:
            branchings.add(i)
    return leafs, branchings


def set_node_attrs(graph, nodes):
    """
    Set node attributes by extracting values from "graph".

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    nodes : list
        List of nodes to set attributes.

    Returns
    -------
    dict
        Dictionary in which keys are node ids and values are a dictionary of
        attributes extracted from "swc_dict".

    """
    attrs = dict()
    for i in nodes:
        attrs[i] = {
            "radius": graph.nodes[i]["radius"], "xyz": graph.nodes[i]["xyz"]
        }
    return attrs


def init_edge_attrs(graph, i):
    """
    Initializes edge attribute dictionary with attributes from node "i" which
    is an end point of the edge.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    i : int
        End point of edge and the swc attributes of this node are used to
        initialize the edge attriubte dictionary.

    Returns
    -------
    dict
        Edge attribute dictionary.

    """
    attrs = {
        "length": 0,
        "radius": [graph.nodes[i]["radius"]],
        "xyz": [graph.nodes[i]["xyz"]],
    }
    return attrs


def upd_edge_attrs(graph, attrs, i, j):
    """
    Updates an edge attribute dictionary with attributes of node i.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of edge being updated.
    i : int
        Node of edge whose attributes will be added to "attrs".

    Returns
    -------
    dict
        Edge attribute dictionary.

    """
    attrs["length"] += compute_dist(graph, i, j)
    attrs["radius"].append(graph.nodes[i]["radius"])
    attrs["xyz"].append(graph.nodes[i]["xyz"])


def set_edge_attrs(graph, attrs):
    for e, attrs_e in attrs.items():
        i, j = tuple(e)
        attrs_e["xyz"][0] = graph.nodes[i]["xyz"]
        attrs_e["xyz"][-1] = graph.nodes[j]["xyz"]
        attrs[e] = attrs_e
    return attrs


def get_edge_attr(graph, edge, attr):
    """
    Gets the attribute "attr" of "edge".

    Parameters
    ----------
    graph : networkx.Graph
        Graph which "edge" belongs to.
    edge : tuple
        Edge to be queried for its attributes.
    attr : str
        Attribute to be queried.

    Returns
    -------
    Attribute "attr" of "edge"

    """
    return graph.edges[edge][attr]


def to_numpy(attrs):
    """
    Converts edge attributes from a list to NumPy array.

    Parameters
    ----------
    attrs : dict
        Dictionary containing attributes of some edge.

    Returns
    -------
    dict
        Updated edge attribute dictionary.

    """
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    return attrs


# --- Miscellaneous ---
def smooth_branch(graph, attrs, i, j):
    """
    Smoothes a branch then updates "swc_dict" and "edges" with the new xyz
    coordinates of the branch end points. Note that this branch is an edge
    in the irreducible graph being built.

    Parameters
    ----------
    graph : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of branch being smoothed.
    i : int
        End point of branch to be smoothed.
    j : int
        End point of branch to be smoothed.

    Returns
    -------
    dict, dict
        Dictionaries that have been updated with respect to smoothed edges.

    """
    attrs["xyz"] = geometry.smooth_branch(attrs["xyz"], s=2)
    graph.nodes[i]["xyz"] = attrs["xyz"][0]
    graph.nodes[j]["xyz"] = attrs["xyz"][-1]


def compute_path_length(graph):
    """
    Computes the path length of the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose nodes have an attribute called "xyz" which represents
        a 3d coordinate.

    Returns
    -------
    float
        Path length of graph.

    """
    path_length = 0
    for i, j in nx.dfs_edges(graph):
        path_length += compute_dist(graph, i, j)
    return path_length


def compute_dist(graph, i, j):
    """
    Computes Euclidean distance between nodes i and j.

    Parameters
    ----------
    graph : netowrkx.Graph
        Graph containing nodes i and j.
    i : int
        Node.
    j : int
        Node.

    Returns
    -------
    float
        Euclidean distance between i and j.

    """
    return geometry.dist(graph.nodes[i]["xyz"], graph.nodes[j]["xyz"])


def cycle_exists(graph):
    """
    Checks whether a cycle exists in "graph".

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be checked for cycles.

    Returns
    -------
    bool
        Indication of whether there exists a cycle.

    """
    try:
        nx.find_cycle(graph)
        return True
    except nx.exception.NetworkXNoCycle:
        return False


def get_leafs(graph):
    """
    Gets leaf nodes of "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched

    Returns
    -------
    list
        Leaf nodes "graph".

    """
    return [i for i in graph.nodes if graph.degree[i] == 1]


def sample_node(graph):
    """
    Samples a single node from a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Node.

    """
    nodes = list(graph.nodes)
    return sample(nodes, 1)[0]


def get_component(graph, root):
    """
    Gets the connected component corresponding to "root" from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    root : int
        Node that breadth-first search starts from.

    Returns
    -------
    set[int]
        Set of nodes in the connected component corresponding to "root".

    """
    queue = [root]
    visited = set()
    while len(queue):
        i = queue.pop()
        visited.add(i)
        for j in [j for j in graph.neighbors(i) if j not in visited]:
            queue.append(j)
    return visited


def count_components(graph):
    """
    Counts the number of connected components in a graph.

    Paramters
    ---------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------'
    int
        Number of connected components.

    """
    return nx.number_connected_components(graph)


def largest_components(graph, k):
    """
    Finds the "k" largest connected components in "graph".

    Parameters
    ----------
    graph : nx.Graph
        Graph to be searched.
    k : int
        Number of largest connected components to return.

    Returns
    -------
    list
        List where each entry is a random node from one of the k largest
        connected components.

    """
    component_cardinalities = k * [-1]
    node_ids = k * [-1]
    for nodes in nx.connected_components(graph):
        if len(nodes) > component_cardinalities[-1]:
            i = 0
            while i < k:
                if len(nodes) > component_cardinalities[i]:
                    component_cardinalities.insert(i, len(nodes))
                    component_cardinalities.pop(-1)
                    node_ids.insert(i, util.sample_singleton(nodes))
                    node_ids.pop(-1)
                    break
                i += 1
    return node_ids
