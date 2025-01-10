"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Overview
--------
Code that reads and preprocesses neuron fragments stored as swc files, then
constructs a custom graph object called a "FragmentsGraph" from them.

    Graph Construction Algorithm:
        1. Read Neuron Fragments
            to do...

        2. Extract Irreducibles
            to do...

        3. Build FragmentsGraph
            to do...

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from random import sample

import networkx as nx
import numpy as np
from tqdm import tqdm

from deep_neurographs import geometry
from deep_neurographs.utils import swc_util, util


class GraphLoader:
    """
    Class that is used to build an instance of FragmentsGraph.

    """

    def __init__(
        self,
        anisotropy=[1.0, 1.0, 1.0],
        min_size=30,
        node_spacing=1,
        progress_bar=False,
        prune_depth=20,
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

        # Step 1: Read Neuron Fragments
        graph_list = self.reader.load(fragments_pointer)

        # Step: Extract Irreducibles
        irreducibles = self.process_graphs(graph_list)

        # Step 3: Build FragmentsGraph
        fragments_graph = FragmentsGraph(node_spacing=self.node_spacing)
        while len(irreducibles):
            irreducible_set = irreducibles.pop()
            fragments_graph.add_component(irreducible_set)
        return fragments_graph

    def process_graphs(self, graphs_list):
        """
        Processes a list of graphs in parallel and extracts irreducible
        subgraphs from each graph.

        Parameters
        ----------
        graphs_list : List[network.Graph]
            List of graphs to be processed. Each graph is passed to the
            "process_graph" method, which extracts the irreducible subgraphs
            from each graph.

        Returns
        -------
        List[dict]
            List of irreducible subgraphs extracted from the input graphs.

        """
        # Initializations
        if self.progress_bar:
            pbar = tqdm(total=len(graphs_list), desc="Process Graphs")

        # Main
        with ProcessPoolExecutor(max_workers=1) as executor:
            # Assign Processes
            processes = list()
            while graphs_list:
                graph = graphs_list.pop()
                processes.append(
                    executor.submit(self.extract_irreducibles, graph)
                )

            # Store results
            irreducibles = list()
            for process in as_completed(processes):
                result = process.result()
                if result is not None:
                    irreducibles.append(result)
                if self.progress_bar:
                    pbar.update(1)
        return irreducibles

    def extract_irreducibles(self, graph):
        """
        Gets the irreducible subgraph from the input graph.

        Parameters
        ----------
        graph : dict
            Graph that irreducible subgraph is to be extracted from.

        Returns
        -------
        List[dict]
            List of dictionaries such that each is the set of irreducibles
            from the input graph.

        """
        irreducibles = None
        self.prune_branches(graph)
        if compute_path_length(graph) > self.min_size:
            # Extract irreducible nodes
            leafs, branchings = get_irreducible_nodes(graph)
            assert len(leafs) > 0, "No leaf nodes!"

            # Extract irreducible edges
            edges = dict()
            root = None
            for (i, j) in nx.dfs_edges(graph, source=util.sample_once(leafs)):
                # Check for start of irreducible edge
                if root is None:
                    root = i
                    path = [i]
                    xyz_list = [graph.nodes[i]["xyz"]]

                # Check for end of irreducible edge
                path.append(j)
                xyz_list.append(graph.nodes[j]["xyz"])
                if j in leafs or j in branchings:
                    edges[(root, j)] = path
                    if self.smooth_bool:
                        graph = smooth_path(graph, path, xyz_list)
                    root = None

            # Set irreducible attributes
            irreducibles = {
                "leaf": set_node_attrs(graph, leafs),
                "branching": set_node_attrs(graph, branchings),
                "edge": set_edge_attrs(graph, edges),
                "swc_id": graph.graph["swc_id"],
            }
        return irreducibles

    def prune_branches(self, graph):
        """
        Prunes all short branches from "graph". A short branch is a path
        between a leaf and branching node where the path length is less than
        "self.prune_depth".

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


# --- Extract Irreducibles  ---
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
    Set node attributes by extracting values from "swc_dict".

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    nodes : list
        List of node ids to set attributes.

    Returns
    -------
    dict
        Dictionary in which keys are node ids and values are a dictionary of
        attributes extracted from "swc_dict".

    """
    node_attrs = dict()
    for i in nodes:
        node_attrs[i] = {
            "radius": graph.nodes[i]["radius"], "xyz": graph.nodes[i]["xyz"]
        }
    return node_attrs


def set_edge_attrs(graph, edges):
    edge_attrs = dict()
    for edge, path in edges.items():
        # Extract attributes
        radius_list, xyz_list = list(), list()
        for i in path:
            radius_list.append(graph.nodes[i]["radius"])
            xyz_list.append(graph.nodes[i]["xyz"])

        # Set attributes
        edge_attrs[edge] = {
            "length": 1000,
            "radius": np.array(radius_list),
            "xyz": np.array(xyz_list)
        }
    return edge_attrs


# --- Miscellaneous ---
def smooth_path(graph, path, xyz_list):
    """
    Smooths a given path on a graph by applying smoothing to the coordinates
    of the nodes along the path and updating the graph with the smoothed
    coordinates.

    Parameters
    ----------
    graph : networkx.Graph
        Graph containing path to be smoothed.
    path : List[int]
        List of node indices representing the path in the graph.
    xyz_list : List[Tuple[float]]
        List of xyz coordinates of path in the graph to be smoothed.

    Returns
    -------
    networkx.Graph
        Input graph with updated "xyz" attributes for the nodes from the input
        path.

    """
    smoothed_xyz_list = geometry.smooth_branch(np.array(xyz_list), s=2)
    for i, xyz in zip(path, smoothed_xyz_list):
        graph.nodes[i]["xyz"] = xyz
    return graph


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
