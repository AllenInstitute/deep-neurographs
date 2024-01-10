"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with graphs.

"""

import networkx as nx

from deep_neurographs import swc_utils, utils


def extract_irreducible_graph(swc_dict, prune=True, prune_depth=16):
    graph = swc_utils.to_graph(swc_dict)
    leafs, junctions = get_irreducibles(graph)
    irreducible_edges, leafs = extract_irreducible_edges(
        graph, leafs, junctions, swc_dict, prune=prune, prune_depth=prune_depth
    )
    if prune:
        irreducible_edges, junctions = check_irreducibility(
            junctions, irreducible_edges
        )
    return leafs, junctions, irreducible_edges


def get_irreducibles(graph):
    leafs = []
    junctions = []
    for i in graph.nodes:
        if graph.degree[i] == 1:
            leafs.append(i)
        elif graph.degree[i] > 2:
            junctions.append(i)
    return leafs, junctions


def extract_irreducible_edges(
    graph, leafs, junctions, swc_dict, prune=True, prune_depth=16
):
    root = None
    irreducible_edges = dict()
    for (i, j) in nx.dfs_edges(graph, source=leafs[0]):
        # Check start of path is valid
        if root is None:
            root = i
            edge = _init_edge(swc_dict=swc_dict, node=i)
            path_length = 0

        # Add to path
        edge["radius"].append(swc_dict["radius"][j])
        edge["xyz"].append(swc_dict["xyz"][j])
        path_length += 1

        # Check whether to end path
        if j in leafs or j in junctions:
            if prune and path_length <= prune_depth:
                condition1 = j in leafs and root in junctions
                condition2 = root in leafs and j in junctions
                if condition1 or condition2:
                    leafs.remove(j if condition1 else root)
                else:
                    irreducible_edges[(root, j)] = edge
            else:
                irreducible_edges[(root, j)] = edge
            root = None
    return irreducible_edges, leafs


def check_irreducibility(junctions, irreducible_edges):
    graph = nx.Graph()
    graph.add_edges_from(irreducible_edges.keys())
    nx.set_edge_attributes(graph, irreducible_edges)
    for j in junctions:
        if j not in graph.nodes:
            junctions.remove(j)
        elif graph.degree[j] == 2:
            # Get join edges
            nbs = list(graph.neighbors(j))
            edge1 = graph.get_edge_data(j, nbs[0])
            edge2 = graph.get_edge_data(j, nbs[1])
            edge = join_edges(edge1, edge2)

            # Update irreducible edges
            junctions.remove(j)
            irreducible_edges = utils.remove_key(
                irreducible_edges, (j, nbs[0])
            )
            irreducible_edges = utils.remove_key(
                irreducible_edges, (j, nbs[1])
            )
            irreducible_edges[tuple(nbs)] = edge

            graph.remove_edge(j, nbs[0])
            graph.remove_edge(j, nbs[1])
            graph.remove_node(j)
            graph.add_edge(*tuple(nbs), xyz=edge["xyz"], radius=edge["radius"])
            if graph.degree[nbs[0]] > 2:
                junctions.append(nbs[0])

            if graph.degree[nbs[1]] > 2:
                junctions.append(nbs[1])

    return irreducible_edges, junctions


def join_edges(edge1, edge2):
    # Last point in edge1 must connect to first point in edge2
    if edge1["xyz"][0] == edge2["xyz"][0]:
        edge1 = reverse_edge(edge1)
    elif edge1["xyz"][-1] == edge2["xyz"][-1]:
        edge2 = reverse_edge(edge2)
    elif edge1["xyz"][0] == edge2["xyz"][-1]:
        edge1 = reverse_edge(edge1)
        edge2 = reverse_edge(edge2)
    edge = {
        "xyz": edge1["xyz"] + edge2["xyz"][1:],
        "radius": edge1["radius"] + edge2["radius"],
    }
    return edge


def reverse_edge(edge):
    edge["xyz"].reverse()
    edge["radius"].reverse()
    return edge


def _init_edge(swc_dict=None, node=None):
    edge = {"radius": [], "xyz": []}
    if node is not None:
        edge["radius"].append(swc_dict["radius"][node])
        edge["xyz"].append(swc_dict["xyz"][node])
    return edge


def get_edge_attr(graph, edge, attr):
    edge_data = graph.get_edge_data(*edge)
    return edge_data[attr]


def is_leaf(graph, i):
    nbs = [j for j in graph.neighbors(i)]
    return True if len(nbs) == 1 else False
