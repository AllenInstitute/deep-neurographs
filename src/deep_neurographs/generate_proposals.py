"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module used to generate edge proposals.

"""

BUFFER = 36


def run(neurograph, query_id, query_xyz, radius):
    """
    Generates edge proposals for node "query_id" in "neurograph" by finding
    candidate points on distinct connected components near "query_xyz".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph built from swc files.
    query_id : int
        Node id of the query node.
    query_xyz : tuple[float]
        (x,y,z) coordinates of the query node.
    radius : float
        Maximum Euclidean distance between end points of edge proposal.

    Returns
    -------
    list
        Best edge proposals generated from "query_node".

    """
    proposals = dict()
    query_swc_id = neurograph.nodes[query_id]["swc_id"]
    for xyz in neurograph.query_kdtree(query_xyz, radius):
        # Check whether xyz is contained (if applicable)
        if not neurograph.is_contained(xyz, buffer=36):
            continue

        # Check whether proposal is valid
        edge = neurograph.xyz_to_edge[tuple(xyz)]
        swc_id = neurograph.edges[edge]["swc_id"]
        if swc_id != query_swc_id and swc_id not in proposals.keys():
            proposals[swc_id] = tuple(xyz)

        # Check whether to stop
        if len(proposals) >= neurograph.proposals_per_leaf:
            break

    return list(proposals.values())


def is_valid(neurograph, i, filter_doubles):
    """
    Determines whether is a valid node to generate proposals from. A node is
    considered valid if it is contained in "self.bbox" (if applicable) and is
    not contained in a doubled connected component (if applicable).

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
    is_double = True if swc_id in neurograph.doubles else False
    is_contained = neurograph.is_contained(i, buffer=BUFFER)
    return False if not is_contained or is_double else True
