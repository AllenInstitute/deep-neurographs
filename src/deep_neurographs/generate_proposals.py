"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module used to generate edge proposals.

"""

import numpy as np

from deep_neurographs import geometry

ENDPOINT_DIST = 10


def run(neurograph, search_radius, complex_bool=True):
    """
    Generates proposals emanating from "leaf".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that proposals will be generated for.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool, optional
        Indication of whether to generate complex proposals. The default is
        True.

    Returns
    -------
    NeuroGraph
        Graph containing leaf that may have been updated.

    """
    connections = dict()
    for leaf in neurograph.leafs:
        neurograph, connections = run_on_leaf(
            neurograph, connections, leaf, search_radius, complex_bool
        )
    # neurograph.filter_nodes()
    return neurograph


def run_on_leaf(neurograph, connections, leaf, search_radius, complex_bool):
    """
    Generates proposals emanating from "leaf".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing leaf.
    connections : dict
        Dictionary that tracks which connected components are connected by a
        proposal. The keys are a frozenset of the pair of swc ids and values
        are the corresponding proposal ids.
    leaf : int
        Leaf node that proposals are to be generated from.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool
        Indication of whether to generate complex proposals.

    Returns
    -------
    NeuroGraph
        Graph containing leaf that may have been updated.
    dict
        Updated "connections" dictionary with information about proposals that
        were added to "neurograph".

    """
    leaf_swc_id = neurograph.nodes[leaf]["swc_id"]
    for xyz in get_candidates(neurograph, leaf, search_radius):
        # Get connection
        neurograph, node = get_conection(neurograph, leaf, xyz, search_radius)
        if not complex_bool and neurograph.degree[node] > 1:
            continue

        # Check whether already connection exists
        pair_id = frozenset((leaf_swc_id, neurograph.nodes[node]["swc_id"]))
        if pair_id in connections.keys():
            proposal = connections[pair_id]
            dist_1 = neurograph.dist(leaf, node)
            dist_2 = neurograph.proposal_length(proposal)
            if dist_1 < dist_2:
                i, j = tuple(proposal)
                neurograph.nodes[i]["proposals"].remove(j)
                neurograph.nodes[j]["proposals"].remove(i)
                del neurograph.proposals[proposal]
                del connections[pair_id]
            else:
                continue

        # Add proposal
        neurograph.add_proposal(leaf, node)
        connections[pair_id] = frozenset({leaf, node})
    return neurograph, connections


def get_candidates(neurograph, leaf, search_radius):
    """
    Generates proposals for node "leaf" in "neurograph" by finding candidate
    xyz points on distinct connected components nearby.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph built from swc files.
    leaf : int
        Leaf node that proposals are to be generated from.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.

    Returns
    -------
    list
        Proposals generated from "leaf".

    """
    candidates = dict()
    dists = dict()
    leaf_xyz = neurograph.nodes[leaf]["xyz"]
    for xyz in neurograph.query_kdtree(leaf_xyz, search_radius):
        swc_id = neurograph.xyz_to_swc(xyz)
        if swc_id != neurograph.nodes[leaf]["swc_id"]:
            if swc_id not in candidates.keys():
                candidates[swc_id] = tuple(xyz)
                dists[swc_id] = geometry.dist(leaf_xyz, xyz)
            elif geometry.dist(leaf_xyz, xyz) < dists[swc_id]:
                candidates[swc_id] = tuple(xyz)
                dists[swc_id] = geometry.dist(leaf_xyz, xyz)
    return get_best_candidates(neurograph, candidates, dists)


def get_best_candidates(neurograph, candidates, dists):
    if len(candidates) > neurograph.proposals_per_leaf:
        worst = None
        for key, d in dists.items():
            if worst is None:
                worst = key
            elif dists[key] > dists[worst]:
                worst = key
        del candidates[worst]
        del dists[worst]
        return get_best_candidates(neurograph, candidates, dists)
    else:
        return list(candidates.values())


def get_conection(neurograph, leaf, xyz, search_radius):
    edge = neurograph.xyz_to_edge[xyz]
    node, d = get_closer_endpoint(neurograph, edge, xyz)
    if d > ENDPOINT_DIST:
        # or neurograph.dist(leaf, node) > search_radius:
        attrs = neurograph.get_edge_data(*edge)
        idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
        node = neurograph.split_edge(edge, attrs, idx)
    return neurograph, node


def is_valid(neurograph, i, filter_doubles):
    """
    Determines whether is a valid node to generate proposals from. A node is
    considered valid if it is not contained in a doubled connected component
    (if applicable).

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph built from swc files.
    i : int
        Node to be validated.
    filter_doubles : bool
        Indication of whether to prevent proposals from being connected to a
        doubled connected component.

    Returns
    -------
    bool
        Indication of whether node is valid.

    """
    if filter_doubles:
        neurograph.upd_doubles(i)
    swc_id = neurograph.nodes[i]["swc_id"]
    return True if swc_id in neurograph.doubles else False


# -- utils --
def get_closer_endpoint(neurograph, edge, xyz):
    i, j = tuple(edge)
    d_i = geometry.dist(neurograph.nodes[i]["xyz"], xyz)
    d_j = geometry.dist(neurograph.nodes[j]["xyz"], xyz)
    return (i, d_i) if d_i < d_j else (j, d_j)
