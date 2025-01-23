"""
Created on Friday June 28 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running seeded inference with a model that classifies edge
proposals.

"""
from copy import deepcopy

import networkx as nx

from deep_neurographs.machine_learning import feature_generation
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.utils import (
    graph_util as gutil,
    img_util,
    ml_util,
    util
)

BATCH_SIZE_PROPOSALS = 1000
CHUNK_SHAPE = (256, 256, 256)


def run(
    neurograph,
    model_path,
    model_type,
    img_path,
    labels_path,
    proposals,
    search_radius,
    batch_size_proposals=BATCH_SIZE_PROPOSALS,
    confidence_threshold=0.7,
    output_dir=None,
):
    # Initializations
    graph = neurograph.copy_graph()
    model = ml_util.load_model(model_type, model_path)

    # Run seeded inference until graphs are fully grown
    accepts = list()
    for swc_id, root in neurograph.soma_ids.items():
        seed_graph = init_seed_graph(neurograph, proposals, root)
        batch_accepts = predict(
            neurograph,
            seed_graph,
            graph,
            model,
            model_type,
            img_path,
            labels_path,
            proposals,
            search_radius,
        )
        accepts.extend(batch_accepts)

        # merge accepted proposals
        # upd seed graph
        #   --> delete rejects
        #   --> merge accepts
        #   --> attempt to grow

    # Run graph-based infererence

    # check whether any proposals remain
    # --> call run_without_seeds
    return accepts, neurograph


def init_seed_graph(neurograph, proposals, root):
    """
    Initializes a seed graph from a root by growing a subgraph of "neurograph"
    from "root".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that seed graph is built from.
    proposals : list
        Proposals to be classified.
    root : int
        Node id.

    Returns
    -------
    networkx.Graph
        Seed graph.

    """
    n_proposals_added = 0
    nodes = gutil.get_component(neurograph, root)
    seed_graph = neurograph.subgraph(nodes)
    seed_graph.graph["proposals"] = set()
    for i in nodes:
        # Add proposals as edges to seed_graph
        for j in neurograph.nodes[i]["proposals"]:
            proposal = frozenset((i, j))
            swc_id = neurograph.nodes[j]["swc_id"]
            if proposal in proposals and not neurograph.is_soma(swc_id):
                seed_graph = add_component(neurograph, seed_graph, j)
                seed_graph.graph["proposals"].add(proposal)
                n_proposals_added += 1

        # Check whether to continue
        if n_proposals_added > BATCH_SIZE_PROPOSALS:
            break
    return seed_graph


def predict(
    neurograph,
    seed_graph,
    graph,
    model,
    model_type,
    img_path,
    labels_path,
    proposals,
    search_radius,
):
    features = feature_generation.run(
        neurograph,
        model_type,
        search_radius,
        img_path,
        labels_path=labels_path,
        proposals=proposals,
    )
    # dataset = ml_util.init_dataset(neurograph, features, model_type)


def build_from_xyz(
    neurograph, labels_path, chunk_origin, chunk_shape=CHUNK_SHAPE, n_hops=1
):
    swc_ids = img_util.get_chunk_labels(labels_path, chunk_origin, chunk_shape)
    seed_neurograph = build_seed_neurograph(neurograph, swc_ids)
    return seed_neurograph


def build_seed_neurograph(neurograph, swc_ids):
    seed_neurograph = NeuroGraph()
    for nodes in nx.connected_components(neurograph):
        i = util.sample_singleton(nodes)
        swc_id = int(neurograph.nodes[i]["swc_id"])
        if swc_id in swc_ids:
            seed_neurograph.update(neurograph.subgraph(nodes))
            seed_neurograph.add_swc_id(swc_id)
    return seed_neurograph


def build_from_boundary(neurograph, pred_neurograph, boundary_components):
    new_bdd = list()
    for c in boundary_components:
        pred_neurograph, bdd = expand_boundary(neurograph, pred_neurograph, c)
        new_bdd.extend(bdd)
    return pred_neurograph, new_bdd


def expand_boundary(neurograph, pred_neurograph, component):
    bdd = list()
    for i in component:
        for j in neurograph.nodes[i]["proposals"]:
            # Extract proposals info
            edge = frozenset({i, j})
            swc_id = int(neurograph.nodes[j]["swc_id"])

            # Add component to graph
            if swc_id not in pred_neurograph.swc_ids:
                c = gutil.get_component(neurograph, j)
                pred_neurograph.add_swc_id(swc_id)
                pred_neurograph = ingest_subgraph(
                    neurograph, pred_neurograph, c
                )
                bdd.append(c)

            # Add proposal to graph
            pred_neurograph.proposals[edge] = deepcopy(
                neurograph.proposals[edge]
            )
            pred_neurograph.nodes[i]["proposals"].add(j)
            pred_neurograph.nodes[j]["proposals"].add(i)

    return pred_neurograph, bdd


def ingest_subgraph(neurograph_1, neurograph_2, node_subset):
    # Add structural components
    subgraph = nx.subgraph(neurograph_1, node_subset)
    neurograph_2.add_nodes_from(node_subset)
    neurograph_2.add_edges_from(subgraph.edges)

    # Add attributes
    for node in node_subset:
        neurograph_2.nodes[node].update(neurograph_1.nodes[node])
        neurograph_2.nodes[node]["proposals"] = set()

    for edge in subgraph.edges:
        neurograph_2.edges[edge].update(neurograph_1.edges[edge])

    return neurograph_2


# --- util ---
def add_component(neurograph, seed_graph, root):
    """
    Adds the connected component that "root" is part of to "seed_graph".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that contains the node "root" and subgraph "seed_graph".
    seed_graph : networkx.Graph
        Seed graph that is a subgraph of "neurograph".
    root : int
        Node id.

    Returns
    -------
    networkx.Graph
        Updated seed graph.

    """
    nodes = gutil.get_component(neurograph, root)
    graph = neurograph.subgraph(nodes)
    seed_graph.add_nodes_from(nodes)
    seed_graph.add_edge_from(list(graph.edges))
    return seed_graph
