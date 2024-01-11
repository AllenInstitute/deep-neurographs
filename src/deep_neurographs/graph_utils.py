"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines that extract the irreducible components of a graph.

"""

from copy import deepcopy
from random import sample

import networkx as nx
import numpy as np

from deep_neurographs import geometry_utils, swc_utils, utils


def get_irreducibles(swc_dict, prune=True, depth=16, smooth=True):
    """
    Gets irreducible components of the graph stored in "swc_dict". The
    irreducible components consist of the leaf and junction nodes along with
    the edges among this set of nodes.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    prune : True
        Indication of whether to prune short branches.
    depth : int
        Path length that determines whether a branch is short.
    smooth : bool
        Indication of whether to smooth each branch.

    Returns
    -------
    leafs : set
        Nodes with degreee 1.
    junctions : set
        Nodes with degree > 2.
    edges : dict
        Set of edges connecting nodes in leafs and junctions. The keys are
        pairs of nodes connected by an edge and values are a dictionary of
        attributes.

    """
    # Initializations
    dense_graph = swc_utils.to_graph(swc_dict)
    if prune:
        dense_graph = prune_short_branches(dense_graph, depth)

    # Extract irreducibles
    leafs, junctions = get_irreducible_nodes(dense_graph)
    source = sample(leafs, 1)[0]
    root = None
    edges = dict()
    nbs = dict()
    for (i, j) in nx.dfs_edges(dense_graph, source=source):
        # Check if start of path is valid
        if root is None:
            root = i
            attrs = __init_edge_attrs(swc_dict, i)

        # Visit j
        attrs = __upd_edge_attrs(swc_dict, attrs, j)
        if j in leafs or j in junctions:
            if smooth:
                swc_dict, edges = __smooth_branch(
                    swc_dict, attrs, edges, nbs, root, j
                )
            else:
                edges[(root, j)] = attrs
            nbs = append_value(nbs, root, j)
            nbs = append_value(nbs, j, root)
            root = None
    return leafs, junctions, edges


def get_irreducible_nodes(graph):
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
    remove_nodes = []
    for leaf in get_leafs(graph):
        remove_nodes.extend(inspect_branch(graph, leaf, depth))
    graph.remove_nodes_from(remove_nodes)
    return graph


def inspect_branch(graph, leaf, depth):
    path = [leaf]
    for (i, j) in nx.dfs_edges(graph, source=leaf, depth_limit=depth):
        if graph.degree(j) > 2:
            return path
        elif graph.degree(j) == 2:
            path.append(j)
    return []


def get_leafs(graph):
    return [i for i in graph.nodes if graph.degree[i] == 1]


def __smooth_branch(swc_dict, attrs, edges, nbs, root, j):
    attrs["xyz"] = geometry_utils.smooth_branch(np.array(attrs["xyz"]))
    swc_dict, edges = upd_xyz(swc_dict, attrs, edges, nbs, root, 0)
    swc_dict, edges = upd_xyz(swc_dict, attrs, edges, nbs, j, -1)
    edges[(root, j)] = attrs
    return swc_dict, edges


def upd_xyz(swc_dict, attrs, edges, nbs, i, start_or_end):
    if i in nbs.keys():
        for j in nbs[i]:
            key = (i, j) if (i, j) in edges.keys() else (j, i)
            edges = upd_branch_endpoint(
                edges, key, swc_dict["xyz"][i], attrs["xyz"][start_or_end]
            )
    swc_dict["xyz"][i] = attrs["xyz"][start_or_end]
    return swc_dict, edges


def append_value(my_dict, key, value):
    if key in my_dict.keys():
        my_dict[key].append(value)
    else:
        my_dict[key] = [value]
    return my_dict


def upd_branch_endpoint(edges, key, old_xyz, new_xyz):
    if all(edges[key]["xyz"][0] == old_xyz):
        edges[key]["xyz"][0] = new_xyz
    else:
        edges[key]["xyz"][-1] = new_xyz
    return edges


# -- attribute utils --
def __init_edge_attrs(swc_dict, i):
    return {"radius": [swc_dict["radius"][i]], "xyz": [swc_dict["xyz"][i]]}


def __upd_edge_attrs(swc_dict, attrs, i):
    attrs["radius"].append(swc_dict["radius"][i])
    attrs["xyz"].append(swc_dict["xyz"][i])
    return attrs


def get_edge_attr(graph, edge, attr):
    edge_data = graph.get_edge_data(*edge)
    return edge_data[attr]
