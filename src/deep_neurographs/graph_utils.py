"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for working with graphs.

"""

import networkx as nx


def get_irreducibles(graph):
    leafs = []
    junctions = []
    for i in graph.nodes:
        if graph.degree[i] == 1:
            leafs.append(i)
        elif graph.degree[i] > 2:
            junctions.append(i)
    return leafs, junctions


def extract_irreducible_graph(swc_dict):
    # Initialize graph
    graph = nx.Graph()
    graph.add_edges_from(zip(swc_dict["id"][1:], swc_dict["pid"][1:]))

    # Extract nodes
    leafs, junctions = get_irreducibles(graph)
    irreducible_nodes = set(leafs + junctions)

    # Extract edges
    irreducible_edges = dict()
    cur_root = None
    cur_edge = _init_edge()
    for (i, j) in nx.dfs_edges(graph, source=leafs[0]):
        # Check start of path is valid
        if cur_root is None:
            assert i in irreducible_nodes, "Logical error in ingest_swc()"
            cur_root = i
            cur_edge = _init_edge(swc_dict=swc_dict, node=i)

        # Add to path
        cur_edge["radius"].append(swc_dict["radius"][j])
        cur_edge["xyz"].append(swc_dict["xyz"][j])

        # Check whether to end path
        if j in irreducible_nodes:
            irreducible_edges[(cur_root, j)] = cur_edge
            cur_root = None

    return leafs, junctions, irreducible_edges


def _init_edge(swc_dict=None, node=None):
    edge = {"radius": [], "xyz": []}
    if node is not None:
        edge["radius"].append(swc_dict["radius"][node])
        edge["xyz"].append(swc_dict["xyz"][node])
    return edge
 

def get_edge_attr(graph, edge, attr):
    edge_data = graph.get_edge_data(*edge)
    return edge_data[attr]
