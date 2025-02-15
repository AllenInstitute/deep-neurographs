"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for both training graph neural networks and running inference
with them.

"""

from collections import defaultdict

import networkx as nx
import torch

from deep_neurographs.utils import graph_util as gutil, ml_util, util

GNN_DEPTH = 2


# --- Tensor Operations ---
def get_inputs(data, device="cpu"):
    """
    Extracts input data for a graph-based model and optionally moves it to a
    GPU.

    Parameters
    ----------
    data : torch_geometric.data.HeteroData
        A data object with the following attributes:
            - x_dict: Dictionary of node features for different node types.
            - edge_index_dict: Dictionary of edge indices for edge types.
             - edge_attr_dict: Dictionary of edge attributes for edge types.
    device : str, optional
        Target device for the data, 'cuda' for GPU and 'cpu' for CPU. The
        default is "cpu".

    Returns
    --------
    tuple:
        Tuple containing the following:
        - x (dict): Node features dictionary.
        - edge_index (dict): Edge indices dictionary.
        - edge_attr (dict): Edge attributes dictionary.

    """
    data.to(device)
    return data.x_dict, data.edge_index_dict, data.edge_attr_dict


# --- Batch Generation ---
def get_batch(
    fragments_graph, proposals, batch_size, flagged_proposals=set()
):
    """
    Gets a batch for training or inference that consist of a computation graph
    and list of proposals. Note: queue contains tuples that consist of a node
    id and distance from proposal.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph that contains proposals to be classified.
    proposals : list
        Proposals to be classified as accept or reject.
    batch_size : int
        Maximum number of proposals in the computation graph.
    flagged_proposals : List[frozenset], optional
        List of proposals that are part of a large connected component in the
        proposal induced subgraph of "fragments_graph". The default is None

    Returns
    -------
    dict
        Batch which consists of a subset of "proposals" and the computation
        graph if the model type is a gnn.

    """
    # Helpers
    def visit_proposal(p):
        batch["graph"].add_edge(i, j)
        batch["proposals"].add(p)
        proposals.remove(p)
        queue.append((j, 0))

    # Main
    batch = reset_batch()
    visited = set()
    while len(proposals) > 0 and len(batch["proposals"]) < batch_size:
        root = tuple(util.sample_once(proposals))
        queue = [(root[0], 0), (root[1], 0)]
        while len(queue) > 0:
            # Visit node's neighbors
            i, d = queue.pop()
            visited.add(i)
            for j in fragments_graph.neighbors(i):
                if (i, j) not in batch["graph"].edges:
                    batch["graph"].add_edge(i, j)

            # Visit node's proposals
            for j in fragments_graph.nodes[i]["proposals"]:
                p = frozenset({i, j})
                if p in proposals and p in flagged_proposals:
                    for q in fragments_graph.proposal_connected_component(p):
                        visit_proposal(q)
                        q_0, q_1 = tuple(q)
                        if q_0 not in visited:
                            queue.append((q_0, 0))
                        if q_1 not in visited:
                            queue.append((q_1, 0))
                elif p in proposals:
                    visit_proposal(p)

            # Update queue
            if len(batch["proposals"]) < batch_size:
                nbhd_i = fragments_graph.neighbors(i)
                for j in [j for j in nbhd_i if j not in visited]:
                    n_proposals = len(fragments_graph.nodes[j]["proposals"])
                    d_j = min(d + 1, -n_proposals)
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
    """
    Computes the number of proposals associated with each node.

    Parameters
    ----------
    proposals : List[frozenset]
        A list of pairs of nodes that represent a proposal in a fragments
        graph.

    Returns
    -------
    defaultdict
        Dictionary where keys are node identifiers and values are the count of
        proposals each node appears in.

    """
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
