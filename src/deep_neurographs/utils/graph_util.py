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

from collections import defaultdict
from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from random import sample
from tqdm import tqdm

import ast
import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, img_util, swc_util, util


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
        segmentation_path=None,
        smooth_bool=True,
        somas_path=None,
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
            FragmentsGraph. The default is 30.0 microns.
        node_spacing : int, optional
            Sampling rate for nodes in FragmentsGraph. Every "node_spacing"
            node is retained. The default is 1.
        prune_depth : int, optional
            Branches with length less than "prune_depth" microns are pruned.
            The default is 20.0 microns.
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
        # Instance attributes
        self.merges_dict = dict()  # key: swc_id, value: soma xyz
        self.min_size = min_size
        self.node_spacing = node_spacing
        self.prune_depth = prune_depth
        self.smooth_bool = smooth_bool
        self.verbose = verbose

        # Load somas
        if segmentation_path and somas_path:
            self.load_somas(segmentation_path, somas_path)

    def load_somas(self, segmentation_path, somas_path):
        # Process soma locations
        driver = "neuroglancer_precomputed"
        reader = img_util.TensorStoreReader(segmentation_path, driver)
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for xyz_str in util.read_txt(somas_path):
                xyz = ast.literal_eval(xyz_str)
                voxel = img_util.to_voxels(xyz, (0.748, 0.748, 1.0))
                threads.append(executor.submit(reader.read_voxel, voxel, xyz))

            # Store results
            id_to_xyz = defaultdict(list)
            for thread in as_completed(threads):
                xyz, seg_id = thread.result()
                if seg_id != 0:
                    id_to_xyz[str(seg_id)].append(xyz)

        # Detect merges - ids that intersect with 2+ somas
        self.merges_dict = {k: v for k, v in id_to_xyz.items() if len(v) > 1}
        print("# Merges Detected:", len(self.merges_dict))

    def remove_merges(self, swc_dicts):
        # Break fragments
        updates = list()
        n_breaks = 0
        for i, swc_dict in enumerate(swc_dicts):
            if swc_dict["swc_id"] in self.merges_dict:
                somas_xyz = self.merges_dict[swc_dict["swc_id"]]
                swc_dict_list = break_fragment(swc_dict, somas_xyz)
                updates.append((i, swc_dict_list))
                n_breaks += len(swc_dict) - 1

        # Update swc_dicts
        updates.reverse()
        for i, swc_dict_list in updates:
            swc_dicts.pop(i)
            swc_dicts.extend(swc_dict_list)
        return swc_dicts

    def get_irreducibles(self, swc_dicts):
        """
        Processes a list of swc dictionaries in parallel and extracts the
        components of the irreducible subgraph from each. Note: this routine
        also breaks fragments that intersect multiple somas if soma locations
        are provided.

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
        # Remove detected merges (if applicable)
        if len(self.merges_dict) > 0:
            swc_dicts = self.remove_merges(swc_dicts)

        # Extract irreducibles
        desc = "Extract Graphs"
        pbar = tqdm(total=len(swc_dicts), desc=desc) if self.verbose else None
        with ProcessPoolExecutor() as executor:
            # Assign Processes
            processes = [None] * len(swc_dicts)
            for i, swc_dict in tqdm(enumerate(swc_dicts), desc="assign process"):
                processes[i] = executor.submit(
                    self.extract_irreducibles, swc_dict
                )
                swc_dict[i] = None
            print("processes assigned")

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
        # Initializations
        if "graph" in swc_dict:
            graph = swc_dict["graph"]
        else:
            graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)

        # Main
        self.prune_branches(graph)
        if path_length(graph, self.min_size) > self.min_size:
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
                    length += dist(graph, i, j)
                    if graph.degree(j) == 2:
                        branch.append(j)
                    elif graph.degree(j) > 2:
                        deleted_nodes.extend(branch)
                        graph.remove_nodes_from(branch)
                        break

                    # Check whether to stop
                    if length > self.prune_depth:
                        break


# --- Break Merged Fragments ---
def break_fragment(swc_dict, somas_xyz):
    graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)
    if len(somas_xyz) == 2:
        # Find points on fragment
        xyz_1, xyz_2 = tuple(somas_xyz)
        node_1 = find_closest_node(graph, xyz_1)
        node_2 = find_closest_node(graph, xyz_2)

        # Find nodes to remove
        nodes = set()
        path = nx.shortest_path(graph, source=node_1, target=node_2)
        for i, node in enumerate(path):
            if graph.degree(node) > 2:
                nodes.add(node)

        # Update graph
        remove_nodes(graph, nodes)
        swc_dict_list = list()
        for i, nodes in enumerate(map(set, nx.connected_components(graph))):
            is_soma = True if node_1 in nodes or node_2 in nodes else False
            swc_dict_i = {
                "graph": graph.subgraph(nodes).copy(),
                "is_soma": is_soma,
                "swc_id": swc_dict["swc_id"] + "." + str(i)
            }
            swc_dict_list.append(swc_dict_i)
    else:
        swc_dict_list = [swc_dict]
    return swc_dict_list


def remove_nodes(graph, nodes, max_radius=7):
    remove = set()
    for root in nodes:
        queue = [(root, 0)]
        visited = set()
        while len(queue) > 0:
            # Visit node
            i, dist_i = queue.pop()
            visited.add(i)

            # Update queue
            for j in graph.neighbors(i):
                dist_j = dist_i + dist(graph, i, j)
                if j not in visited and dist_j <= max_radius:
                    queue.append((j, dist_j))
        remove = remove.union(visited)
    graph.remove_nodes_from(nodes)


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
    j : int
        ...

    Returns
    -------
    dict
        Edge attribute dictionary.

    """
    attrs["length"] += dist(graph, i, j)
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


def cycle_exists(graph):
    """
    Checks if the given graph has a cycle.

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be searched.

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


def dist(graph, i, j):
    """
    Computes Euclidean distance between nodes i and j.

    Parameters
    ----------
    graph : netowrkx.Graph
        Graph containing nodes i and j.
    i : int
        Node ID.
    j : int
        Node ID.

    Returns
    -------
    float
        Euclidean distance between nodes i and j.

    """
    return geometry_util.dist(graph.nodes[i]["xyz"], graph.nodes[j]["xyz"])


def find_closest_node(graph, xyz):
    best_dist = np.inf
    best_node = None
    for i in graph.nodes:
        cur_dist = geometry_util.dist(xyz, graph.nodes[i]["xyz"])
        if cur_dist < best_dist:
            best_dist = cur_dist
            best_node = i
    return best_node


def find_shortest_path(graph, source, target):
    """
    Finds the shortest path between two edges in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    source : int
        Edge from which to start the shortest path search.
    target : int
        Edge to which the shortest path should end.

    Returns
    -------
    List[int]
        Nodes representing the shortest path from the source node to the
        target node. Note: this path includes all nodes in source and target
        edges.

    """
    # Find shortest path
    path = nx.shortest_path(graph, source=source[0], target=target[0])
    print(path, source, target)

    # Check if both nodes in source are in path
    if path[1] != source[1]:
        path.insert(0, source[1])

    # Check if both nodes in target are in path
    if len(path) == 2:
        path.insert(-1, target[1])
    elif path[-2] != target[1]:
        path.insert(-1, target[1])
    return path


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
        Nodes in the connected component corresponding to "root".

    """
    queue = [root]
    visited = set()
    while len(queue):
        i = queue.pop()
        visited.add(i)
        for j in [j for j in graph.neighbors(i) if j not in visited]:
            queue.append(j)
    return visited


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


def path_length(graph, max_length=np.inf):
    """
    Computes the path length of the given graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph whose nodes have an attribute called "xyz" which represents
        a 3d coordinate.
    max_length : float
        ...

    Returns
    -------
    float
        Path length of graph.

    """
    path_length = 0
    for i, j in nx.dfs_edges(graph):
        path_length += dist(graph, i, j)
        if path_length > max_length:
            break
    return path_length


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
