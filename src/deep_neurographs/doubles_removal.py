"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments from a NeuroGraph.

"""

from deep_neurographs import utils
import networkx as nx


def run(neurograph, min_size, max_size, node_spacing, output_dir=None):
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
    cnt = 1
    t0, t1 = utils.init_timers()
    components = list(nx.connected_components(neurograph))
    n_components = len(components)
    chunk_size = int(n_components * 0.02)
    
    # Main
    doubles_cnt = 0
    neurograph.init_kdtree()
    not_doubles = set()
    for i, nodes in enumerate(components):
        # Determine whether to inspect fragment
        swc_id = get_swc_id(neurograph, nodes)
        if swc_id not in not_doubles:
            xyz_arr = inspect_component(neurograph, nodes)
            upper = len(xyz_arr) * node_spacing < max_size
            lower = len(xyz_arr) * node_spacing > min_size
            if upper and lower:
                not_double_id = is_double(neurograph, xyz_arr, swc_id)
                if not_double_id:
                    doubles_cnt += 1
                    if output_dir:
                        neurograph.to_swc(output_dir, nodes, color="1.0 0.0 0.0")
                    neurograph = remove_component(neurograph, nodes, swc_id)
                    not_doubles.add(not_double_id)

        # Update progress bar
        if i > cnt * chunk_size:
            cnt, t1 = utils.report_progress(
                i + 1, n_components, chunk_size, cnt, t0, t1
            )
    print("\n# Doubles detected:", doubles_cnt)


def is_double(neurograph, fragment, swc_id_i):
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
    # Compute projections
    hits = dict()
    for xyz_i in fragment:
        for xyz_j in neurograph.query_kdtree(xyz_i, 5):
            try:
                swc_id_j = neurograph.xyz_to_swc(xyz_j)
                if swc_id_i != swc_id_j:
                    hits = utils.append_dict_value(hits, swc_id_j, 1)
            except:
                pass

    # Check criteria
    if len(hits) > 0:
        swc_id_j = utils.find_best(hits)
        percent_hit = len(hits[swc_id_j]) / len(fragment)
    else:
        percent_hit = 0
    return swc_id_j if swc_id_j is not None and percent_hit > 0.5 else None


# --- utils ---
def get_swc_id(neurograph, nodes):
    """
    Gets the swc id corresponding to "nodes".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing "nodes".
    nodes : list[int]
        Nodes to be checked.

    Returns
    -------
    str
        swc id of "nodes".

    """
    i = utils.sample_singleton(nodes)
    return neurograph.nodes[i]["swc_id"]


def inspect_component(neurograph, nodes):
    """
    Determines whether to inspect component for doubles.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be searched.
    nodes : iterable
        Nodes that comprise a connected component.

    Returns
    -------
    numpy.ndarray or list
        Array containing xyz coordinates of nodes.

    """
    if len(nodes) == 2:
        i, j = tuple(nodes)
        return neurograph.edges[i, j]["xyz"]
    else:
        return []


def remove_component(neurograph, nodes, swc_id):
    """
    Removes "nodes" from "neurograph".

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
