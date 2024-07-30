"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training graph neural networks.

"""

import networkx as nx
import numpy as np
import torch

GRAPH_BATCH_SIZE = 10000

def toCPU(tensor):
    """
    Moves tensor from GPU to CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor.

    Returns
    -------
    None

    """
    return tensor.detach().cpu().tolist()


def get_inputs(data, model_type):
    if "Hetero" in model_type:
        x = data.x_dict
        edge_index = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        return x, edge_index, edge_attr_dict
    else:
        x = data.x
        edge_index = data.edge_index
        return x, edge_index


def to_tensor(my_list):
    """
    Converts a list to a tensor with contiguous memory.

    Parameters
    ----------
    my_list : list
        List to be converted into a tensor.

    Returns
    -------
    torch.Tensor
        Tensor.

    """
    arr = np.array(my_list, dtype=np.int64).tolist()
    return torch.Tensor(arr).t().contiguous().long()


def init_line_graph(edges):
    """
    Initializes a line graph from a list of edges.

    Parameters
    ----------
    edges : list
        List of edges.

    Returns
    -------
    networkx.Graph
        Line graph generated from a list of edges.

    """
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return nx.line_graph(graph)


def get_batches(graph, proposals, batch_size=GRAPH_BATCH_SIZE):
    """
    Gets batches during inference that consist of a computation graph and
    list of proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be partitioned into batches
    proposals : list
        Proposals to be classified as accept or reject.
    batch_size : int, optional
        Maximum number of nodes in the computation graph. The default is the
        global variable "GRAPH_BATCH_SIZE".

    Returns
    -------
    networkx.Graph
        Computation graph that gnn will perform inference with.
    list
        Proposals contained within computation graph.

    """
    batch = reset_batch()
    graph.add_edges_from(proposals)
    for cc in nx.connected_components(graph):
        # Determine whether to add component
        cc_graph = graph.subgraph(cc)
        cc_proposals = list_proposals_in_graph(cc_graph, proposals)
        if len(cc_proposals) > 0 and len(cc_proposals) < batch_size:
            # Determine whether to start new batch
            cur_batch_size = batch["graph"].number_of_nodes()
            if cur_batch_size + cc_graph.number_of_nodes() > batch_size:
                print("yield", len(cc_proposals))
                yield batch
                batch = reset_batch()

            # Update batch
            batch["graph"] = nx.union(batch["graph"], cc_graph)
            batch["proposals"].extend(cc_proposals)
        elif len(cc_proposals) > batch_size:
            print("Too many proposals in connected component!")
            print("# proposals:", len(cc_proposals))
            print("")
    yield batch


def reset_batch():
    """
    Resets the current batch.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        Batch that consists of a graph and list of proposals.

    """
    return {"graph": nx.Graph(), "proposals": list()}


def list_proposals_in_graph(graph, proposals):
    """
    Lists the proposals that are edges in "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    proposals : list[frozenset]
        Proposals of interest.

    Returns
    -------
    list
        Proposals that are edges in "graph".
    """
    return [frozenset(e) for e in graph.edges if frozenset(e) in proposals]
