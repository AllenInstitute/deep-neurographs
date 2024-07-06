"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a NeuroGraph.

"""
import sys
from time import time

import networkx as nx
import numpy as np

from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.geometry import dist, path_length

COLOR = "1.0 0.0 0.0"
MAX_DEPTH = 16


# --- Doubles Removal ---
def remove_doubles(neurograph, max_size, node_spacing, output_dir=None):
    """
    Removes connected components from "neurgraph" that are likely to be a
    double.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be searched for doubles.
    max_size : int
        Maximum size of connected components to be searched.
    node_spacing : int
        Expected distance in microns between nodes in "neurograph".
    output_dir : str or None, optional
        Directory that doubles will be written to. The default is None.

    Returns
    -------
    NeuroGraph
        Graph with doubles removed.

    """
    # Initializations
    deleted = set()
    neurograph.init_kdtree()
    nodes = list(nx.connected_components(neurograph))

    # Main
    cnt = 1
    t0, t1 = utils.init_timers()
    for i, idx in enumerate(np.argsort([len(c) for c in nodes])):
        # Determine whether to check component
        node = utils.sample_singleton(nodes[idx])
        swc_id = neurograph.nodes[node]["swc_id"]
        if len(nodes[idx]) == 2 and swc_id not in deleted:
            fragment_size = len(neurograph.edges[tuple(nodes[idx])]["xyz"])
            if fragment_size * node_spacing < max_size:
                # Check doubles criteria
                n_points = len(neurograph.edges[tuple(nodes[idx])]["xyz"])
                hits = compute_hits(neurograph, tuple(nodes[idx]), swc_id)
                if check_doubles_criteria(hits, n_points, swc_id):
                    if output_dir:
                        neurograph.to_swc(output_dir, nodes[idx], color=COLOR)
                    neurograph = delete_nodes(neurograph, nodes[idx], swc_id)
                    deleted.add(swc_id)

        # Update progress bar
        if i >= cnt * len(nodes) * 0.02:
            cnt, t1 = utils.report_progress(
                i + 1, len(nodes), len(nodes) * 0.02, cnt, t0, t1
            )
    print("\n# Doubles detected:", len(deleted))


def compute_hits(neurograph, edge, query_id):
    hits = dict()
    for i, xyz in enumerate(neurograph.edges[edge]["xyz"]):
        # Compute projections
        best_id = None
        best_dist = np.inf
        for hit_xyz in neurograph.query_kdtree(xyz, 15):
            try:
                hit_id = neurograph.xyz_to_swc(hit_xyz)
                if hit_id != query_id:
                    if dist(hit_xyz, xyz) < best_dist:
                        best_dist = dist(hit_xyz, xyz)
                        best_id = hit_id
            except:
                pass

        # Store best
        if best_id is not None:
            hits = utils.append_dict_value(hits, best_id, best_dist)

        # Check whether to stop
        if i == 8:
            if len(hits) == 0:
                return hits
    return hits


def check_doubles_criteria(hits, n_points, swc_id):
    """
    Determines whether the connected component corresponding to "root" is a
    double of another connected component.

    Paramters
    ---------
    neurograph : NeuroGraph
        Graph to be searched for doubles.
    fragment : numpy.ndarray
        Array containing xyz coordinates corresponding to some fragment (i.e.
        connected component in neurograph).
    swc_id_i : str
        swc id corresponding to fragment.

    Returns
    -------
    str or None
        Indication of whether connected component is a double. If True, the
        swc_id of the main fragment (i.e. non doubles) is returned. Otherwise,
        the value None is returned to indicate that query fragment is not a
        double.

    """
    for hit_id, dists in hits.items():
        if len(dists) > 5:
            percent_hit = len(dists) / n_points
            dist_std = np.std(dists)
            if percent_hit > 0.5 and dist_std < 2:
                return True
            elif percent_hit > 0.75 and dist_std < 2.5:
                return True
            """
            elif len(dists) > 10:
                dists = np.array(dists)
                if np.mean(dists[0:10]) < 5:
                    print(swc_id, hit_id)
                    print(percent_hit, dist_std)
                    print(
                        len(dists),
                        np.sum(dists < 5),
                        np.mean(dists),
                        np.mean(dists[0:10]),
                        np.sum(dists < 5) / n_points
                    )
                    print("")
            """
    return False


def delete_nodes(neurograph, nodes, swc_id):
    """
    Deletes "nodes" from "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that contains "nodes".
    nodes : list[int]
        Nodes to be removed.
    swc_id : str
        swc id corresponding to nodes which comprise a connected component in
        "neurograph".

    Returns
    -------
    NeuroGraph
        Graph with nodes removed.

    """
    i, j = tuple(nodes)
    neurograph = remove_xyz_entries(neurograph, i, j)
    neurograph.remove_nodes_from([i, j])
    neurograph.leafs.remove(i)
    neurograph.leafs.remove(j)
    neurograph.swc_ids.remove(swc_id)
    return neurograph


def remove_xyz_entries(neurograph, i, j):
    """
    Removes dictionary entries from "neurograph.xyz_to_edge" corresponding to
    the edge {i, j}.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be updated.
    i : int
        Node in "neurograph".
    j : int
        Node in "neurograph".

    Returns
    -------
    NeuroGraph
        Updated graph.

    """
    for xyz in neurograph.edges[i, j]["xyz"]:
        del neurograph.xyz_to_edge[tuple(xyz)]
    return neurograph


def upd_hits(hits, key, value):
    """
    Updates "hits" by adding ("key", "value") if this item does not exist.
    Otherwise, checks if "value" is less than "hits[key"]".

    Parameters
    ----------
    hits : dict
        Stores swd_ids of fragments that are within a certain distance a query
        fragment along with the corresponding distances.
    key : str
        swc id of some fragment.
    value : float
        Distance in microns that fragment corresponding to "key" is from the
        query fragment.

    Returns
    -------
    dict
        Updated version of hits.

    """
    if key in hits.keys():
        if value < hits[key]:
            hits[key] = value
    else:
        hits[key] = value
    return hits


# --- Trim Passings ---
def trim_passings(neurograph):
    # Initializations
    cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = int(len(neurograph.leafs) * 0.02)

    # Main
    n_endpoints_trimmed = 0
    for leaf in neurograph.leafs:
        if leaf in neurograph.nodes:
            hits = search_along_branch(neurograph, leaf)
            if len(hits) == 1:
                neurograph = simple_trim(neurograph, leaf, hits)
                n_endpoints_trimmed += 1
            elif len(hits) > 1:
                print("complex trim")
    print("# hits:", n_endpoints_trimmed)


def search_along_branch(neurograph, leaf):
    hits = dict()
    swc_id_leaf = neurograph.nodes[leaf]["swc_id"]
    for xyz_leaf, radius in get_branch(neurograph, leaf):
        for xyz in neurograph.query_kdtree(xyz_leaf, radius + 1):
            try:
                swc_id = neurograph.xyz_to_swc(xyz)
                if swc_id != swc_id_leaf:
                    hits = utils.append_dict_value(hits, swc_id, xyz)
            except:
                pass
    return keep_passings(hits) if len(hits) > 0 else hits


def keep_passings(hits):
    rm_keys = list()
    for swc_id, xyz_coords in hits.items():
        if path_length(xyz_coords) < 5:
            rm_keys.append(swc_id)
    return utils.remove_items(hits, rm_keys)


def simple_trim(neurograph, leaf, hits):
    swc_id, xyz_coords = unpack_dict(hits)
    i, j = get_edge(neurograph, xyz_coords)
    if neurograph.is_leaf(i) or neurograph.is_leaf(j):
        # Check for significant difference in radii
        i = get_endpoint(neurograph, leaf, (i, j))
        radius_leaf = neurograph.nodes[leaf]["radius"]
        radius_edge = neurograph.nodes[i]["radius"]
        
        if radius_leaf < radius_edge - 1 or radius_leaf / radius_edge < 0.75:
            print("Case 1:")
            print("trimming:", neurograph.nodes[leaf]["swc_id"])
            print("not trimming:", swc_id)
            neurograph = trim_from_leaf(neurograph, leaf, xyz_coords)
        elif radius_edge < radius_leaf - 1 or radius_edge / radius_leaf < 0.75:
            print("Case 2:")
            print("trimming:", neurograph.nodes[leaf]["swc_id"])
            print("not trimming:", swc_id)
            neurograph = trim_from_leaf(neurograph, i, xyz_coords)
        else:
            # Determine smaller fragment
            leaf_component_size = len(gutils.get_component(neurograph, leaf))
            edge_component_size = len(gutils.get_component(neurograph, i))
            if leaf_component_size < edge_component_size:
                print("Case 3a:")
                print("trimming:", neurograph.nodes[leaf]["swc_id"])
                print("not trimming:", swc_id)
                neurograph = trim_from_leaf(neurograph, leaf, xyz_coords)
            else:
                print("Case 3b:")
                print("trimming:", neurograph.nodes[leaf]["swc_id"])
                print("not trimming:", swc_id)
                neurograph = trim_from_leaf(neurograph, i, xyz_coords)
    return neurograph


def get_endpoint(neurograph, leaf, edge):
    i, j = tuple(edge)
    d_i = dist(neurograph.nodes[i]["xyz"], neurograph.nodes[leaf]["xyz"])
    d_j = dist(neurograph.nodes[j]["xyz"], neurograph.nodes[leaf]["xyz"])
    return i if d_i < d_j else j


def trim_from_leaf(neurograph, leaf, xyz_coords):
    # Initializations
    j = list(neurograph.neighbors(leaf))[0]
    xyz_coords = set([tuple(xyz) for xyz in xyz_coords])

    # Determine points to trim
    idx = 0
    zipped_branch = get_branch(neurograph, leaf)
    while len(xyz_coords) > 0:
        for xyz_query, radius in zipped_branch:
            for xyz in neurograph.query_kdtree(xyz_query, radius + 0.5):
                if tuple(xyz) in xyz_coords:
                    xyz_coords.remove(tuple(xyz))
        idx += 1
        if idx > len(neurograph.edges[leaf, j]["xyz"]):
            break
    xyz_coords = neurograph.oriented_edge((leaf, j), leaf)
    idx = min(len(xyz_coords), idx + 2)

    # Trim points
    if path_length(xyz_coords[idx::]) > 15:
        print(f"--> trimmed {round(path_length(xyz_coords[0:idx]), 2)} microns \n")
        neurograph = trim(neurograph, leaf, j, xyz_coords, idx)
    else:
        print("--> deleting\n")
        neurograph.remove_node(leaf)
        if neurograph.degree[j] == 0:
            neurograph.remove_node(j)
    return neurograph


def trim(neurograph, leaf, j, xyz_coords, idx):
    e = (leaf, j)
    neurograph.nodes[leaf]["xyz"] = xyz_coords[idx]
    neurograph.edges[leaf, j]["xyz"] = xyz_coords[idx::]
    neurograph.edges[e]["radius"] = neurograph.edges[e]["radius"][idx::]
    return neurograph    

 
# --- utils ---
def get_branch(neurograph, i):
    """
    Gets the xyz coordinates of the branch emanating from "i".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing node "i".
    i : int
        Leaf node in "neurograph".

    Returns
    -------
    numpy.ndarray
        xyz coordinates of branch emanating from "i".

    """
    j = list(neurograph.neighbors(i))[0]
    xyz_coords = neurograph.oriented_edge((i, j), i)
    radii = neurograph.oriented_edge((i, j), i, key="radius")
    n = max(len(xyz_coords), MAX_DEPTH)
    return zip(xyz_coords[0:n], radii[0:n])


def get_branch_avg_radii(neurograph, leaf):
    """
    Gets the average radii of the branch emanating from "leaf".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing "leaf".
    leaf : int
        Node id of leaf node.

    Returns
    -------
    float
        Average radii of the branch emanating from "leaf".

    """
    j = list(neurograph.neighbors(leaf))[0]
    return np.mean(neurograph.oriented_edge((leaf, j), leaf, key="radius"))


def unpack_dict(my_dict):
    return list(my_dict.items())[0]


def get_edge(neurograph, xyz_coords):
    hits = dict()
    for xyz in xyz_coords:
        try:
            edge = neurograph.xyz_to_edge[tuple(xyz)]
            hits = utils.append_dict_value(hits, edge, 1)
        except:
            pass
    return tuple(utils.find_best(hits))
