"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines that extract the irreducible components of a graph.


Terminology
------------

Leaf: a node with degree 1.

Junction: a node with degree > 2.

Irreducibles: the irreducibles of a graph G=(V,E) consists of 1) leaf nodes
V_l, 2) junction nodes, and 3) edges connecting (1) and (2).

Branch: the sequence of nodes between two irreducible nodes.

"""

from random import sample

import networkx as nx
import numpy as np

from deep_neurographs import geometry_utils, swc_utils, utils


def get_irreducibles(swc_dict, swc_id=None, prune=True, depth=16, smooth=True):
    """
    Gets irreducible components of the graph stored in "swc_dict". The
    irreducible components consist of the leaf and junction nodes along with
    the edges among this set of nodes.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    swc_id : str, optional
        Filename of swc which is used to run this routine with
        multiprocessing. The default is None.
    prune : bool, optional
        Indication of whether to prune short branches. The default is True.
    depth : int, optional
        Path length that determines whether a branch is short. The default is
        16.
    smooth : bool, optional
        Indication of whether to smooth each branch. The default is True.

    Returns
    -------
    dict
        Irreducibles stored in a dictionary where key-values are type of
        irreducible (i.e. leaf, junction, or edge) and corresponding set of
        all irreducibles from the graph of that type.

    """
    # Build dense graph
    dense_graph = swc_utils.to_graph(swc_dict)
    if prune:
        dense_graph = prune_short_branches(dense_graph, depth)

    # Extract nodes
    leafs, junctions = get_irreducible_nodes(dense_graph, swc_dict)
    if len(leafs) == 0:
        return False, None

    # Extract edges
    edges = dict()
    nbs = dict()
    root = None
    for (i, j) in nx.dfs_edges(dense_graph, source=sample(leafs, 1)[0]):
        # Check if start of path is valid
        if root is None:
            root = i
            attrs = init_edge_attrs(swc_dict, i)

        # Visit j
        attrs = upd_edge_attrs(swc_dict, attrs, j)
        if j in leafs or j in junctions:
            attrs = set_edge_attrs(attrs)
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
    leafs = set_node_attrs(swc_dict, leafs)
    junctions = set_node_attrs(swc_dict, junctions)
    irreducibles = {"leafs": leafs, "junctions": junctions, "edges": edges}
    return swc_id, irreducibles


def get_irreducible_nodes(graph, swc_dict):
    """
    Gets irreducible nodes (i.e. leafs and junctions) of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    leafs : set
        Nodes with degreee 1.
    junctions : set
        Nodes with degree > 2.

    """
    leafs = set()
    junctions = set()
    for i in graph.nodes:
        if graph.degree[i] == 1:
            leafs.add(i)
        elif graph.degree[i] > 2:
            junctions.add(i)
    return leafs, junctions


def prune_short_branches(graph, depth):
    """
    Prunes all short branches from "graph". A short branch is a path between a
    leaf and junction node with a path length smaller than depth.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched
    depth : int
        Path length that determines whether a branch is short.

    Returns
    -------
    graph : networkx.Graph
        Graph with short branches pruned.

    """
    remove_nodes = []
    for leaf in get_leafs(graph):
        remove_nodes.extend(inspect_branch(graph, leaf, depth))
    graph.remove_nodes_from(remove_nodes)
    return graph


def inspect_branch(graph, leaf, depth):
    """
    Determines whether the branch emanating from "leaf" should be pruned.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    leaf : int
        Leaf node being inspected to determine whether it is the endpoint of
        a short branch that should be pruned.
    depth : int
        Path length that determines whether a branch is short.

    Returns
    -------
    list
        If branch is short, then the list of nodes in the branch is returned.
        Otherwise, an empty list is returned.

    """
    path = [leaf]
    for (i, j) in nx.dfs_edges(graph, source=leaf, depth_limit=depth):
        if graph.degree(j) > 2:
            return path
        elif graph.degree(j) == 2:
            path.append(j)
    return []


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
    """
    attrs["xyz"] = geometry_utils.smooth_branch(attrs["xyz"], s=10)
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
    swc_dict : dict
        Updated with new xyz coordinates.
    edges : dict
        Updated with new xyz coordinates.

    """
    if i in nbs.keys():
        for j in nbs[i]:
            key = (i, j) if (i, j) in edges.keys() else (j, i)
            edges = upd_endpoint_xyz(
                edges, key, swc_dict["xyz"][i], attrs["xyz"][endpoint]
            )
    swc_dict["xyz"][i] = attrs["xyz"][endpoint]
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
    edges : dict
        Updated with new xyz coordinates.

    """
    if all(edges[key]["xyz"][0] == old_xyz):
        edges[key]["xyz"][0] = new_xyz
    else:
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
    return {"radius": [swc_dict["radius"][i]], "xyz": [swc_dict["xyz"][i]]}


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
    attrs : dict
        Edge attribute dictionary.

    """
    attrs["radius"].append(swc_dict["radius"][i])
    attrs["xyz"].append(swc_dict["xyz"][i])
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

 
def set_edge_attrs(attrs):
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    return attrs

 
def set_node_attrs(swc_dict, nodes):
    attrs = dict()
    for i in nodes:
        attrs[i] = {
            "radius": swc_dict["radius"][i], "xyz": swc_dict["xyz"][i]
        }
    return attrs


def upd_node_attrs(swc_dict, leafs, junctions, i):
    upd_attrs = {"radius": swc_dict["radius"][i], "xyz": swc_dict["xyz"][i]}
    if i in leafs:
        leafs[i] = upd_attrs
    else:
        junctions[i] = upd_attrs
    return leafs, junctions
