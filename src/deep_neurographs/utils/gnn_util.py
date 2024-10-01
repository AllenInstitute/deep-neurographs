"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for training graph neural networks.

"""

from collections import defaultdict

import networkx as nx
import numpy as np
import torch

from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import util

GNN_DEPTH = 2


# --- Tensor Operations ---
def get_inputs(data, device=None):
    # Extract inputs
    x = data.x_dict
    edge_index = data.edge_index_dict
    edge_attr = data.edge_attr_dict

    # Move to gpu (if applicable)
    if "cuda" in device and torch.cuda.is_available():
        x = toGPU(x, device)
        edge_index = toGPU(edge_index, device)
        edge_attr = toGPU(edge_attr, device)    
    return x, edge_index, edge_attr


def toGPU(tensor_dict, device):
    """
    Moves dictionary of tensors from CPU to GPU.

    Parameters
    ----------
    tensor_dict : dict
        Tensor to be moved to GPU.

    Returns
    -------
    None

    """
    return {k: tensor.to("cuda") for k, tensor in tensor_dict.items()}


def toCPU(tensor):
    """
    Moves tensor from GPU to CPU.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be moved to CPU.

    Returns
    -------
    None

    """
    return tensor.detach().cpu().tolist()


def toTensor(my_list):
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


# --- Batch Generation ---
def get_batch(graph, proposals, batch_size):
    """
    Gets a batch for training or inference that consist of a computation graph
    and list of proposals. Note: queue contains tuples that consist of a node
    id and distance from proposal.

    Parameters
    ----------
    graph : NeuroGraph
        Graph that contains proposals
    proposals : list
        Proposals to be classified as accept or reject.
    batch_size : int
        Maximum number of proposals in the computation graph.

    Returns
    -------
    dict
        Batch which consists of a subset of "proposals" and the computation
        graph if the model type is a gnn.

    """
    batch = reset_batch()
    visited = set()
    while len(proposals) > 0 and len(batch["proposals"]) < batch_size:
        root = tuple(util.sample_once(proposals))
        queue = [(root[0], 0), (root[1], 0)]
        while len(queue) > 0:
            # Visit node
            i, d = queue.pop()
            for j in graph.neighbors(i):
                if (i, j) not in batch["graph"].edges:
                    batch["graph"].add_edge(i, j)

            for p in graph.nodes[i]["proposals"]:
                if frozenset({i, p}) in proposals:
                    batch["graph"].add_edge(i, p)
                    batch["proposals"].add(frozenset({i, p}))
                    proposals.remove(frozenset({i, p}))
                    queue.append((p, 0))
            visited.add(i)

            # Update queue
            if len(batch["proposals"]) < batch_size:
                for j in [j for j in graph.neighbors(i) if j not in visited]:
                    d_j = min(d + 1, -len(graph.nodes[j]["proposals"]))
                    if d_j <= GNN_DEPTH:
                        queue.append((j, d + 1))
    return batch


def get_train_batch(graph, proposals, batch_size):
    """
    Gets a batch for training or inference that consist of a computation graph
    and list of proposals.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be partitioned into batches
    proposals : list
        Proposals to be classified as accept or reject.
    batch_size : int, optional
        Maximum number of nodes in the computation graph. The default is the
        global variable "BATCH_SIZE".

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
        cc_graph = graph.subgraph(cc).copy()
        cc_proposals = proposals_in_graph(cc_graph, proposals)

        # Check whether to sample from graph
        while len(cc_proposals) + len(batch["proposals"]) > batch_size:
            node_proposal_cnt = get_node_proposal_cnt(cc_proposals)
            batch = extract_subgraph_batch(
                cc_graph,
                cc_proposals,
                batch,
                batch_size,
                node_proposal_cnt,
            )
            if len(batch["proposals"]) >= batch_size:
                n_proposals = 0
                for p in batch["proposals"]:
                    if tuple(p) not in batch["graph"].edges:
                        n_proposals += 1
                        batch["graph"].add_edges_from([p])

                yield batch
                batch = reset_batch()

        # Check whether to return or update batch
        if len(cc_proposals) > 0:
            batch["graph"].add_edges_from(cc_graph.edges)
            batch["proposals"].extend(cc_proposals)

    # Yield remaining batch if not empty
    if len(batch["proposals"]) > 0:

        n_proposals = 0
        for p in batch["proposals"]:
            if tuple(p) not in batch["graph"].edges:
                n_proposals += 1
                batch["graph"].add_edges_from([p])
        yield batch


def extract_subgraph_batch(
    graph, proposals, batch, batch_size, node_proposal_cnt
):
    # Extract batch via bfs
    remove_nodes = set()
    n_proposals_added = 0
    for i, j in nx.bfs_edges(graph, source=gutil.sample_node(graph)):
        # Visit edge
        batch["graph"].add_edge(i, j)

        # Check if edge is proposal
        p = frozenset({i, j})
        if p in proposals:
            n_proposals_added += 1
            batch["proposals"].append(p)
            proposals.remove(p)

        # Update node_proposal_cnt
        node_proposal_cnt[i] -= 1
        node_proposal_cnt[j] -= 1
        if node_proposal_cnt[i] == 0:
            remove_nodes.add(i)
        if node_proposal_cnt[j] == 0:
            remove_nodes.add(j)

        # Check whether batch is full
        if len(batch["proposals"]) >= batch_size:
            break

    # Yield batch
    graph.remove_nodes_from(remove_nodes)
    return batch


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
    return {"graph": nx.Graph(), "proposals": set()}


# --- Miscellaneous ---
def proposals_in_graph(graph, proposals):
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
    return set([e for e in map(frozenset, graph.edges) if e in proposals])


def get_node_proposal_cnt(proposals):
    node_proposal_cnt = defaultdict(int)
    for i, j in proposals:
        node_proposal_cnt[i] += 1
        node_proposal_cnt[j] += 1
    return node_proposal_cnt


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
