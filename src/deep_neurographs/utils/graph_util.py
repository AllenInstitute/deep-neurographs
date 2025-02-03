"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Overview
--------
Code that loads and preprocesses neuron fragments stored as swc files, then
constructs a custom graph object called a "FragmentsGraph".

Note: We use the term "branch" to refer to a path in a graph from a branching
      node to a leaf.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from random import sample
from tqdm import tqdm

import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, swc_util, util


class GraphLoader:
    """
    Class that loads SWC files and constructs a FragmentsGraph instance from
    the data.

    """

    def __init__(
        self,
        anisotropy=[1.0, 1.0, 1.0],
        min_size=30.0,
        node_spacing=1,
        prune_depth=20.0,
        smooth_bool=True,
        verbose=False,
    ):
        """
        Builds a FragmentsGraph by reading swc files stored on either the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        anisotropy : List[float], optional
            Image to physical coordinates scaling factors to account for the
            anisotropy of the microscope. The default is [1.0, 1.0, 1.0].
        min_size : float, optional
            Minimum path length of swc files that are loaded into the
            FragmentsGraph. The default is 30.0 (microns).
        node_spacing : int, optional
            Sampling rate for nodes in FragmentsGraph. Every "node_spacing"
            node is retained. The default is 1.
        prune_depth : int, optional
            Branches with length less than "prune_depth" microns are pruned.
            The default is 20.0 microns.
        smooth_bool : bool, optional
            Indication of whether to smooth xyz coordinates from SWC files.
            The default is True.
        verbose : bool, optional
            Indication of whether to display a progress bar while building
            FragmentsGraph. The default is True.

        Returns
        -------
        None

        """
        self.min_size = min_size
        self.node_spacing = node_spacing
        self.prune_depth = prune_depth
        self.smooth_bool = smooth_bool
        self.verbose = verbose

    def get_irreducibles(self, swc_dicts):
        """
        Processes a list of swc dictionaries in parallel and extracts the
        components of the irreducible subgraph from each.

        Parameters
        ----------
        swc_dicts : List[dict]
            List of dictionaries such that each contains the contents of an
            SWC file.

        Returns
        -------
        List[dict]
            List of dictionaries such that each contains the components of the
            irreducible subgraph extracted from each SWC dictionary.

        """
        # Initializations
        if self.verbose:
            pbar = tqdm(total=len(swc_dicts), desc="Extract Graphs")

        # Main
        with ProcessPoolExecutor() as executor:
            # Assign Processes
            i = 0
            processes = [None] * len(swc_dicts)
            while swc_dicts:
                swc_dict = swc_dicts.pop()
                processes[i] = executor.submit(
                    self.extract_irreducibles, swc_dict
                )
                i += 1

            # Store results
            irreducibles = list()
            for process in as_completed(processes):
                result = process.result()
                if result is not None:
                    irreducibles.append(result)
                if self.verbose:
                    pbar.update(1)
        return irreducibles

    def extract_irreducibles(self, swc_dict):
        """
        Gets the components of the irreducible subgraph from a given SWC
        dictionary.

        Parameters
        ----------
        swc_dict : dict
            Contents of an SWC file.

        Returns
        -------
        dict
            Dictionary that each contains the components of an irreducible
            subgraph.

        """
        graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)
        self.prune_branches(graph)
        if compute_path_length(graph, self.min_size) > self.min_size:
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
                "edge": set_edge_attrs(graph, edges, self.node_spacing),
                "swc_id": swc_dict["swc_id"],
                "is_soma": swc_dict["is_soma"],
            }
            return irreducibles
        else:
            return None

    def prune_branches(self, graph):
        """
        Prunes branches with length less than "self.prune_depth" microns.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to be searched.

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
                        break


# --- Irreducibles Extraction ---
def get_irreducible_nodes(graph):
    """
    Gets irreducible nodes (i.e. leafs and branchings) of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    Tuple[set]
        Sets of leaf and branching nodes.

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
    Extracts attributes for each node in the graph.

    Parameters
    ----------
    nodes : List[int]
        Nodes whose attributes are to be extracted from the graph.

    Returns
    -------
    dict
        Dictionary where the keys are node ids and values are dictionaries
        containing the "radius" and "xyz" attributes of the nodes.

    """
    attrs = dict()
    for i in nodes:
        attrs[i] = {
            "radius": graph.nodes[i]["radius"], "xyz": graph.nodes[i]["xyz"]
        }
    return attrs


def init_edge_attrs(graph, i):
    """
    Initializes an attribute dictionary for a single irreducible edge.

    Parameters
    ----------
    graph : networkx.Graph
        Graph containing node i.
    i : int
        Node that is the start of a path between irreducible nodes. The
        attributes of this node are used to initialize the dictionary.

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
    graph : networkx.Graph
        Graph containing nodes i and j.
    attrs : dict
        Edge attribute dictionary to be updated.
    i : int
        Node in the path between irreducible nodes whose attributes will be
        added to the "attrs" dictionary.

    Returns
    -------
    dict
        Edge attribute dictionary.

    """
    attrs["length"] += compute_dist(graph, i, j)
    attrs["radius"].append(graph.nodes[i]["radius"])
    attrs["xyz"].append(graph.nodes[i]["xyz"])


def set_edge_attrs(graph, attrs, node_spacing):
    """
    Sets the edge attributes of a given graph by updating node coordinates and
    resamples points in irreducible path.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that attributes dictionary was built from.
    attrs : dict
        Dictionary where the keys are irreducible edge IDs and values are the
        corresponding attribute dictionaries.
    node_spacing : int
        Sampling rate for nodes in FragmentsGraph. Every "node_spacing"-th node
        is retained.

    Returns
    -------
    dict
        Updated edge attribute dictionary.

    """
    for e in attrs:
        # Update endpoints
        i, j = tuple(e)
        attrs[e]["xyz"][0] = graph.nodes[i]["xyz"]
        attrs[e]["xyz"][-1] = graph.nodes[j]["xyz"]

        # Resample points
        idxs = util.spaced_idxs(len(attrs[e]["xyz"]), node_spacing)
        attrs[e]["radius"] = attrs[e]["radius"][idxs]
        attrs[e]["xyz"] = attrs[e]["xyz"][idxs]
    return attrs


def to_numpy(attrs):
    """
    Converts edge attributes from a list to NumPy array.

    Parameters
    ----------
    attrs : dict
        Edge attribute dictionary.

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
    Smoothes branch then updates "graph" and "attrs" with the new xyz
    coordinates.

    Parameters
    ----------
    graph : dict
        Graph containing branch to be smoothed.
    attrs : dict
        Edge attributes dictionary.
    i : int
        End point of branch to be smoothed.
    j : int
        End point of branch to be smoothed.

    Returns
    -------
    None

    """
    attrs["xyz"] = geometry_util.smooth_branch(attrs["xyz"], s=2)
    graph.nodes[i]["xyz"] = attrs["xyz"][0]
    graph.nodes[j]["xyz"] = attrs["xyz"][-1]


def compute_path_length(graph, max_length=np.inf):
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
        if path_length > max_length:
              break
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
        Euclidean distance between nodes i and j.

    """
    return geometry_util.dist(graph.nodes[i]["xyz"], graph.nodes[j]["xyz"])


def cycle_exists(graph):
    """
    Checks the given graph has a cycle.

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be checked for cycles.

    Returns
    -------
    bool
        Indication of whether graph has a cycle.

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
    List[int]
        Leaf nodes of "graph".

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
    Set[int]
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
    -------
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
    List[int]
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
