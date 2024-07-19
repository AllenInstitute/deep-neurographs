"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines that extract the irreducible components of a graph.


Terminology
------------

Leaf: a node with degree 1.

Junction: a node with degree > 2.

Irreducibles: the irreducibles of a graph consists of 1) leaf nodes,
2) junction nodes, and 3) edges connecting (1) and (2).

Branch: a sequence of nodes between two irreducible nodes.

"""

from random import sample

import networkx as nx
import numpy as np

from deep_neurographs import geometry, swc_utils, utils


def get_irreducibles(
    swc_dict,
    min_size,
    bbox=None,
    prune_connectors=False,
    connector_length=8,
    prune_depth=16,
    trim_depth=0,
    smooth=True,
):
    """
    Gets irreducible components of the graph stored in "swc_dict" by building
    the graph store in the swc_dict and parsing it. In addition, this function
    also calls routines prunes spurious branches and short paths connecting
    branches (i.e. possible merge mistakes).

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    bbox : dict, optional
        ...
    prune_connectors : bool, optional
        Indication of whether to prune short paths connecting branches.
        The default is False.
    prune_depth : float, optional
        Path length microns that determines whether a branch is short and
        should be pruned. The default is 16.
    trim_depth : float, optional
        Depth in microns to trim branch. The default is 0.
    smooth : bool, optional
        Indication of whether to smooth each branch. The default is True.

    Returns
    -------
    list
        List of irreducibles stored in a dictionary where key-values are type
        of irreducible (i.e. leaf, junction, or edge) and corresponding set of
        all irreducibles from the graph of that type.

    """
    # Build dense graph
    swc_dict["idx"] = dict(zip(swc_dict["id"], range(len(swc_dict["id"]))))
    graph, _ = swc_utils.to_graph(swc_dict, set_attrs=True)
    graph = clip_branches(graph, bbox)
    graph, n_nodes_trimmed = prune_branches(
        graph,
        connector_length=connector_length,
        prune_connectors=prune_connectors,
        prune_depth=prune_depth,
        trim_depth=trim_depth,
    )

    # Extract irreducibles
    irreducibles = []
    for node_subset in nx.connected_components(graph):
        if len(node_subset) + n_nodes_trimmed > min_size:
            subgraph = graph.subgraph(node_subset)
            irreducibles_i = __get_irreducibles(subgraph, swc_dict, smooth)
            if irreducibles_i:
                irreducibles.append(irreducibles_i)
    return irreducibles


def clip_branches(graph, bbox):
    """
    Deletes all nodes from "graph" that are not contained in "bbox".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched
    bbox : dict
        Bounding box.

    Returns
    -------
    networkx.Graph
        "graph" with nodes deleted that were not contained in "bbox".

    """
    if bbox:
        delete_nodes = set()
        for i in graph.nodes:
            xyz = utils.to_voxels(graph.nodes[i]["xyz"])
            if not utils.is_contained(bbox, xyz):
                delete_nodes.add(i)
        graph.remove_nodes_from(delete_nodes)
    return graph


def prune_branches(
    graph,
    connector_length=8,
    prune_connectors=False,
    prune_depth=16,
    trim_depth=0,
):
    """
    Prunes spurious branches and short paths connecting branches
    (i.e. possible merge mistakes).

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be pruned.
    connector_length : float, optional
        ...
    prune_connectors : bool, optional
        Indication of whether to prune short paths connecting branches.
        The default is False.
    prune_depth : float, optional
        Path length microns that determines whether a branch is short and
        should be pruned. The default is 16.
    trim_depth : float, optional
        Depth in microns to trim branch. The default is 0.

    Returns
    -------
    networkx.Graph
        Pruned graph.
    int
        Number of nodes that were trimmed.

    """
    # Prune/Trim branches
    assert prune_depth > 0 if prune_connectors else True, "prune_depth == 0"
    if prune_depth > 0:
        graph, n_nodes_trimmed = prune_trim_branches(
            graph, prune_depth, trim_depth
        )

    # Prune connectors
    if prune_connectors:
        graph = prune_short_connectors(graph, connector_length)
    return graph, n_nodes_trimmed


def __get_irreducibles(graph, swc_dict, smooth):
    """
    Gets the irreducible components of "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    swc_dict : dict
        Dictionary that was used to build "graph".
    smooth : bool
        Indication of whether to smooth irreducible edges.

    Returns
    -------
    dict
        Dictionary containing irreducible components of "graph".

    """
    # Extract nodes
    leafs, junctions = get_irreducible_nodes(graph)
    assert len(leafs), "No leaf nodes!"
    if len(leafs) > 0:
        source = sample(leafs, 1)[0]
    else:
        source = sample(junctions, 1)[0]

    # Extract edges
    edges = dict()
    nbs = dict()
    root = None
    for (i, j) in nx.dfs_edges(graph, source=source):
        # Check if start of path is valid
        if root is None:
            root = i
            attrs = init_edge_attrs(swc_dict, root)

        # Visit j
        attrs = upd_edge_attrs(swc_dict, attrs, j)
        if j in leafs or j in junctions:
            attrs = to_numpy(attrs)
            if smooth:
                swc_dict, edges = __smooth_branch(
                    swc_dict, attrs, edges, nbs, root, j
                )
            else:
                edges[(root, j)] = attrs
            nbs = utils.append_dict_value(nbs, root, j)
            nbs = utils.append_dict_value(nbs, j, root)
            root = None

    # Output
    swc_id = swc_dict["swc_id"]
    leafs = set_node_attrs(swc_dict, leafs)
    junctions = set_node_attrs(swc_dict, junctions)
    irreducibles = {
        "leafs": leafs,
        "junctions": junctions,
        "edges": edges,
        "swc_id": swc_id,
    }
    return irreducibles


def get_irreducible_nodes(graph):
    """
    Gets irreducible nodes (i.e. leafs and junctions) of a graph.

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
    junctions = set()
    for i in graph.nodes:
        if graph.degree[i] == 1:
            leafs.add(i)
        elif graph.degree[i] > 2:
            junctions.add(i)
    return leafs, junctions


# --- Refine graph ---
def prune_trim_branches(graph, depth, trim_depth):
    """
    Prunes all short branches from "graph" and trims branchs if applicable. A
    short branch is a path between a leaf and junction node with a path length
    smaller than depth.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    depth : int
        Path length that determines whether a branch is short.

    Returns
    -------
    networkx.Graph
        Graph with short branches pruned.

    """
    remove_nodes = []
    n_nodes_trimmed = 0
    for leaf in get_leafs(graph):
        nodes, n_trimmed = inspect_branch(graph, leaf, depth, trim_depth)
        remove_nodes.extend(nodes)
        n_nodes_trimmed += n_trimmed
    graph.remove_nodes_from(remove_nodes)
    return graph, n_nodes_trimmed


def inspect_branch(graph, leaf, depth, trim_depth):
    """
    Determines whether the branch emanating from "leaf" should be pruned and
    returns nodes that should be pruned. If applicable (i.e. trim_depth > 0),
    trims the branch by "trim_depth" microns.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    leaf : int
        Leaf node being inspected to determine whether it is the endpoint of
        a short branch that should be pruned.
    depth : int
        Path length microns that determines whether a branch is short.
    trim_depth : float
        Depth in microns to trim branch.

    Returns
    -------
    list
        If branch is short, then the list of nodes in the branch is returned.
        Otherwise, an empty list is returned.

    """
    # Check whether to prune
    path = [leaf]
    node_spacing = []
    for (i, j) in nx.dfs_edges(graph, source=leaf, depth_limit=depth):
        node_spacing.append(compute_dist(graph, i, j))
        if graph.degree(j) > 2:
            return path, 0
        elif graph.degree(j) == 2:
            path.append(j)
        elif np.sum(node_spacing) > depth:
            break

    # Check whether to trim
    spacing = np.mean(node_spacing)
    if trim_depth > 0 and graph.number_of_nodes() > 3 * trim_depth / spacing:
        trim_nodes = trim_branch(graph, path, trim_depth)
        return trim_nodes, len(trim_nodes)
    else:
        return [], 0


def trim_branch(graph, path, trim_depth):
    branch_length = 0
    for i in range(1, len(path)):
        xyz_1 = graph.nodes[path[i - 1]]["xyz"]
        xyz_2 = graph.nodes[path[i]]["xyz"]
        branch_length += geometry.dist(xyz_1, xyz_2)
        if branch_length > trim_depth:
            break
    return path[0:i]


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
    Euclidean distance between i and j.

    """
    return geometry.dist(graph.nodes[i]["xyz"], graph.nodes[j]["xyz"])


def prune_short_connectors(graph, length=8):
    """ "
    Prunes shorts paths (i.e. connectors) between junctions nodes and the nbhd
    about the junctions.

    Parameters
    ----------
    graph : netowrkx.Graph
        Graph to be inspected.
    length : int, optional
        Upper bound on the distance that defines a connector path to be
        pruned. The default is 8.

    Returns
    -------
    list[tuple]
        Graph with connectors pruned.
    list[np.ndarray]
        List of xyz coordinates of centroids of connectors.

    """
    junctions = [j for j in graph.nodes if graph.degree[j] > 2]
    pruned_centroids = []
    pruned_nodes = set()
    cnt = 0
    while len(junctions):
        # Search nbhd
        j = junctions.pop()
        junction_nbs = []
        for _, i in nx.dfs_edges(graph, source=j, depth_limit=length):
            if i in junctions:
                junction_nbs.append(i)

        # Store nodes to be pruned
        for nb in junction_nbs:
            connector = list(nx.shortest_path(graph, source=j, target=nb))
            nbhd = set(nx.dfs_tree(graph, source=nb, depth_limit=5))
            centroid = connector[len(connector) // 2]
            if not ignore_connector(graph, centroid, 16 + length // 2):
                pruned_nodes.update(nbhd.union(set(connector)))
                pruned_centroids.append(graph.nodes[centroid]["xyz"])

        if len(junction_nbs) > 0:
            nbhd = set(nx.dfs_tree(graph, source=j, depth_limit=5))
            pruned_nodes.update(nbhd)
            cnt += 1

    # Finish
    graph.remove_nodes_from(list(pruned_nodes))
    return graph


def ignore_connector(graph, root, depth):
    """
    Determines whether the connector is in a region with lots of branching.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    root : int
        Midpoint of connector.

    Returns
    -------
    bool
        Indication of whether connector is in a region with lots of branching.

    """
    n_branching_points = 0
    for i in nx.dfs_tree(graph, source=root, depth_limit=depth):
        if graph.degree[i] > 2:
            n_branching_points += 1
    return True if n_branching_points > 2 else False


def __smooth_branch(swc_dict, attrs, edges, nbs, root, j):
    """
    Smoothes a branch then updates "swc_dict" and "edges" with the new xyz
    coordinates of the branch end points. Note that this branch is an edge
    in the irreducible graph being built.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of edge being smoothed.
    edges : dict
        Dictionary where the keys are edges in irreducible graph and values
        are the corresponding attributes.
    nbs : dict
        Dictionary where the keys are nodes and values are the neighbors.
    root : int
        End point of branch to be smoothed.
    j : int
        End point of branch to be smoothed.

    Returns
    -------
    dict, dict
        Dictionaries that have been updated with respect to smoothed edges.

    """
    attrs["xyz"] = geometry.smooth_branch(attrs["xyz"], s=2)
    swc_dict, edges = upd_xyz(swc_dict, attrs, edges, nbs, root, 0)
    swc_dict, edges = upd_xyz(swc_dict, attrs, edges, nbs, j, -1)
    edges[(root, j)] = attrs
    return swc_dict, edges


def upd_xyz(swc_dict, attrs, edges, nbs, i, endpoint):
    """
    Updates "swc_dict" and "edges" with the new xyz coordinates of the branch
    end points.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of edge being smoothed.
    edges : dict
        Dictionary where the keys are edges in irreducible graph and values
        are the corresponding attributes.
    nbs : dict
        Dictionary where the keys are nodes and values are the neighbors.
    endpoint : int
        End point of branch to be smoothed.

    Returns
    -------
    dict
        Updated with new xyz coordinates.
    dict
        Updated with new xyz coordinates.

    """
    idx = swc_dict["idx"][i]
    if i in nbs.keys():
        for j in nbs[i]:
            key = (i, j) if (i, j) in edges.keys() else (j, i)
            edges = upd_endpoint_xyz(
                edges, key, swc_dict["xyz"][idx], attrs["xyz"][endpoint]
            )
    swc_dict["xyz"][idx] = attrs["xyz"][endpoint]
    return swc_dict, edges


def upd_endpoint_xyz(edges, key, old_xyz, new_xyz):
    """
    Updates "edges" with the new xyz coordinates of the branch
    end points.

    Parameters
    ----------
    edges : dict
        Dictionary where the keys are edges in irreducible graph and values
        are the corresponding attributes.
    key : tuple
        The edge id of the entry in "edges" which needs to be updated.
    old_xyz : numpy.ndarray
        Current xyz coordinate of end point.
    new_xyz : numpy.ndarray
        New xyz coordinate of end point.

    Returns
    -------
    dict
        Updated with new xyz coordinates.

    """
    if all(edges[key]["xyz"][0] == old_xyz):
        edges[key]["xyz"][0] = new_xyz
    elif all(edges[key]["xyz"][-1] == old_xyz):
        edges[key]["xyz"][-1] = new_xyz
    return edges


# -- attribute utils --
def init_edge_attrs(swc_dict, i):
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
    swc_id = swc_dict["swc_id"]
    assert i in swc_dict["idx"].keys(), f"{swc_id} - {i}"
    j = swc_dict["idx"][i]
    return {"radius": [swc_dict["radius"][j]], "xyz": [swc_dict["xyz"][j]]}


def upd_edge_attrs(swc_dict, attrs, i):
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
    swc_id = swc_dict["swc_id"]
    assert i != -1, f"{swc_id} - {i}"
    j = swc_dict["idx"][i]
    attrs["radius"].append(swc_dict["radius"][j])
    attrs["xyz"].append(swc_dict["xyz"][j])
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


def set_node_attrs(swc_dict, nodes):
    """
    Set node attributes by extracting values from "swc_dict".

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
        j = swc_dict["idx"][i]
        attrs[i] = {"radius": swc_dict["radius"][j], "xyz": swc_dict["xyz"][j]}
    return attrs


def upd_node_attrs(swc_dict, leafs, junctions, i):
    """
    Updates node attributes by extracting values from "swc_dict".

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file that contains the smoothed xyz coordinates of
        corresponding to "leafs" and "junctions". Note xyz coordinates are
        smoothed during edge extraction.
    leafs : dict
        Dictionary where keys are leaf node ids and values are attribute
        dictionaries.
    junctions : dict
        Dictionary where keys are junction node ids and values are attribute
        dictionaries.
    i : int
        Node to be updated.

    Returns
    -------
    dict
        Updated dictionary if "i" was contained in "leafs.keys()".
    dict
        Updated dictionary if "i" was contained in "junctions.keys()".

    """
    j = swc_dict["idx"][i]
    upd_attrs = {"radius": swc_dict["radius"][j], "xyz": swc_dict["xyz"][j]}
    if i in leafs:
        leafs[i] = upd_attrs
    else:
        junctions[i] = upd_attrs
    return leafs, junctions


# -- miscellaneous --
def creates_cycle(graph, edge):
    """
    Checks whether adding "edge" to "graph" creates a cycle.

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be checked for cycles.
    edge : tuple
        Edge to be added to "graph"

    Returns
    -------
    bool
        Indication of whether adding "edge" to graph creates a cycle.

    """
    graph.add_edges_from([edge])
    exists = cycle_exists(graph)
    graph.remove_edges_from([edge])
    if exists:
        return True, edge
    else:
        return False, edge


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
    except:
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


def sample_singleton(graph):
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
    component = set()
    while len(queue):
        i = queue.pop()
        component.add(i)
        for j in [j for j in graph.neighbors(i) if j not in component]:
            if (i, j) in graph.edges:
                queue.append(j)
    return component


def count_components(graph):
    """
    Counts the number of connected components in a graph.

    Paramters
    ---------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    Number of connected components.

    """
    return nx.number_connected_components(graph)


def largest_components(neurograph, k):
    """
    Finds the "k" largest connected components in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
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
    for nodes in nx.connected_components(neurograph):
        if len(nodes) > component_cardinalities[-1]:
            i = 0
            while i < k:
                if len(nodes) > component_cardinalities[i]:
                    component_cardinalities.insert(i, len(nodes))
                    component_cardinalities.pop(-1)
                    node_ids.insert(i, utils.sample_singleton(nodes))
                    node_ids.pop(-1)
                    break
                i += 1
    return node_ids
