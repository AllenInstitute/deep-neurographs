"""
Created on Friday June 28 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running seeded inference with a model that classifies edge
proposals.

"""
import networkx as nx

BATCH_SIZE_PROPOSALS = 1000
N_SEED_GRAPHS = 16


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
    seeds = neurograph.soma_ids
    batches = utils.get_batches(list(seeds.keys()), N_SEED_GRAPHS)
    model = ml_utils.load_model(model_type, model_path)

    # Run seeded inference until graphs are fully grown
    print("Running Seeded Inference from Somas...")
    for i, batch in enumerate(batches):
        seed_graphs, swc_ids = build_seed_graphs(neurograph, proposals, batch)
        while len(seed_graphs) > 0:
            # generate features
            # run inference
            # merge accepted proposals
            # upd seed graphs
            #   --> delete rejects
            #   --> merge accepts
            #   --> attempt to grow

    # check whether any proposals remain
    # --> call run_without_seeds

    
def init_seed_graphs(neurograph, proposals, batch):
    seed_graphs = list()
    swc_ids = set()
    for root in batch:
        seed_graph, swc_ids = build_from_node(
            neurograph, proposals, swc_ids, root
        )
        seed_graphs.append(seed_graph)
    return seed_graphs


def build_from_node(neurograph, proposals, swc_ids, root):
    n_proposals_added = 0
    nodes = gutils.get_component(graph, root)
    seed_graph = neurograph.subgraph(nodes)
    seed_graph.graph["proposals"] = set()
    for i in seed_graph.nodes:
        for edge in neurograph.nodes[i]["proposals"]:
            if edge in proposals:
                pass


def build_from_xyz(
    neurograph, labels_path, chunk_origin, chunk_shape=CHUNK_SHAPE, n_hops=1
):
    swc_ids = get_swc_ids(labels_path, chunk_origin, chunk_shape)
    seed_neurograph = build_seed_neurograph(neurograph, swc_ids)
    return seed_neurograph


def build_seed_neurograph(neurograph, swc_ids):
    seed_neurograph = NeuroGraph()
    for nodes in nx.connected_components(neurograph):
        i = utils.sample_singleton(nodes)
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
                c = gutils.get_component(neurograph, j)
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


# --- utils ---
def get_swc_ids(path, xyz, chunk_shape, from_center=True):
    img = utils.open_tensorstore(path, "neuroglancer_precomputed")
    img = utils.read_tensorstore(
        img, xyz, chunk_shape, from_center=from_center
    )
    return set(fastremap.unique(img).astype(int))
