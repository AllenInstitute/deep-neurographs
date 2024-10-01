"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Routines for loading fragments and building a neurograph.


Terminology
------------

Leaf: a node with degree 1.

Junction: a node with degree > 2.

Irreducibles: the irreducibles of a graph consists of 1) leaf nodes,
2) junction nodes, and 3) edges connecting (1) and (2).

Branch: a sequence of nodes between two irreducible nodes.

"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import sample

import networkx as nx
import numpy as np
from tqdm import tqdm

from deep_neurographs import geometry
from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.utils import img_util, swc_util, util

MIN_SIZE = 30
NODE_SPACING = 1
SMOOTH_BOOL = True
PRUNE_DEPTH = 16
TRIM_DEPTH = 0


class GraphLoader:
    """
    Class that is used to build an instance of FragmentsGraph.

    """

    def __init__(
        self,
        anisotropy=[1.0, 1.0, 1.0],
        min_size=MIN_SIZE,
        node_spacing=NODE_SPACING,
        progress_bar=False,
        prune_depth=PRUNE_DEPTH,
        smooth_bool=SMOOTH_BOOL,
        trim_depth=TRIM_DEPTH,
    ):
        """
        Builds a FragmentsGraph by reading swc files stored either on the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        anisotropy : list[float], optional
            Scaling factors applied to xyz coordinates to account for
            anisotropy of microscope. The default is [1.0, 1.0, 1.0].
        min_size : float, optional
            Minimum path length of swc files which are stored as connected
            components in the FragmentsGraph. The default is 30ums.
        node_spacing : int, optional
            Spacing (in microns) between nodes. The default is the global
            variable "NODE_SPACING".
        progress_bar : bool, optional
            Indication of whether to print out a progress bar while building
            graph. The default is True.
        prune_depth : int, optional
            Branches less than "prune_depth" microns are pruned if "prune" is
            True. The default is the global variable "PRUNE_DEPTH".
        smooth_bool : bool, optional
            Indication of whether to smooth branches from swc files. The
            default is the global variable "SMOOTH".
        trim_depth : float, optional
            Maximum path length (in microns) to trim from "branch". The default
            is the global variable "TRIM_DEPTH".

        Returns
        -------
        FragmentsGraph
            FragmentsGraph generated from swc files.

        """
        self.anisotropy = anisotropy
        self.min_size = min_size
        self.node_spacing = node_spacing
        self.progress_bar = progress_bar
        self.prune_depth = prune_depth
        self.smooth_bool = smooth_bool
        self.trim_depth = trim_depth

        self.reader = swc_util.Reader(anisotropy, min_size)

    def run(
        self, fragments_pointer, img_patch_origin=None, img_patch_shape=None
    ):
        """
        Builds a FragmentsGraph by reading swc files stored either on the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        fragments_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentsGraph,
            see "swc_util.Reader" for further documentation.
        img_patch_origin : list[int], optional
            An xyz coordinate which is the upper, left, front corner of the
            image patch that contains the swc files. The default is None.
        img_patch_shape : list[int], optional
            Shape of the image patch which contains the swc files. The default
            is None.

        Returns
        -------
        FragmentsGraph
            FragmentsGraph generated from swc files.

        """
        # Load fragments and extract irreducibles
        self.set_img_bbox(img_patch_origin, img_patch_shape)
        swc_dicts = self.reader.load(fragments_pointer)
        irreducibles = get_irreducibles(
            swc_dicts,
            self.min_size,
            self.img_bbox,
            self.progress_bar,
            self.prune_depth,
            self.smooth_bool,
            self.trim_depth,
        )

        # Build FragmentsGraph
        neurograph = NeuroGraph(node_spacing=self.node_spacing)
        while len(irreducibles):
            irreducible_set = irreducibles.pop()
            neurograph.add_component(irreducible_set)
        return neurograph

    def set_img_bbox(self, img_patch_origin, img_patch_shape):
        """
        Sets the bounding box of an image patch as a class attriubte.

        Parameters
        ----------
        img_patch_origin : tuple[int]
            Origin of bounding box which is assumed to be top, front, left
            corner.
        img_patch_shape : tuple[int]
            Shape of bounding box.

        Returns
        -------
        None

        """
        self.img_bbox = img_util.get_bbox(img_patch_origin, img_patch_shape)


# --- Graph structure extraction ---
def get_irreducibles(
    swc_dicts,
    min_size,
    img_bbox=None,
    progress_bar=True,
    prune_depth=16.0,
    smooth_bool=True,
    trim_depth=0.0,
):
    """
    Gets irreducible components of each graph stored in "swc_dicts" by setting
    up a parellelization scheme that sends each swc_dict to a CPU and calling
    the subroutine "gutil.get_irreducibles".

    Parameters
    ----------
    swc_dicts : list[dict]
        List of dictionaries such that each entry contains the conents of an
        swc file.

    Returns
    -------
    list[dict]
        List of irreducibles stored in a dictionary where key-values are type
        of irreducible (i.e. leaf, junction, or edge) and the corresponding
        set of all irreducibles from the graph of that type.

    """
    with ProcessPoolExecutor() as executor:
        # Assign Processes
        i = 0
        processes = [None] * len(swc_dicts)
        while swc_dicts:
            swc_dict = swc_dicts.pop()
            processes[i] = executor.submit(
                get_component_irreducibles,
                swc_dict,
                min_size,
                img_bbox,
                prune_depth,
                smooth_bool,
                trim_depth,
            )
            i += 1

        # Store results
        irreducibles = list()
        if progress_bar:
            with tqdm(total=len(processes), desc="Extract Graphs") as pbar:
                for process in as_completed(processes):
                    irreducibles.extend(process.result())
                    pbar.update(1)
        else:
            for process in as_completed(processes):
                irreducibles.extend(process.result())
    return irreducibles


def get_component_irreducibles(
    swc_dict,
    min_size,
    img_bbox=None,
    prune_depth=16.0,
    smooth_bool=True,
    trim_depth=0.0,
):
    """
    Gets irreducible components of the graph stored in "swc_dict" by building
    the graph store in the swc_dict and parsing it. In addition, this function
    also calls routines prunes spurious branches and short paths connecting
    branches (i.e. possible merge mistakes).

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    min_size : float
        Minimum cardinality of swc files that are stored in NeuroGraph.
    img_bbox : dict, optional
        Dictionary with the keys "min" and "max" which specify a bounding box
        in the image. The default is None.
    prune_depth : float, optional
        Path length microns that determines whether a branch is short and
        should be pruned. The default is 16.0.
    smooth_bool : bool, optional
        Indication of whether to smooth each branch. The default is True.
    trim_depth : float, optional
        Maximum path length (in microns) to trim from "branch". The default is
        0.0.

    Returns
    -------
    list
        List of irreducibles stored in a dictionary where key-values are type
        of irreducible (i.e. leaf, junction, or edge) and corresponding set of
        all irreducibles from the graph of that type.

    """
    # Build dense graph
    swc_dict["idx"] = dict(zip(swc_dict["id"], range(len(swc_dict["id"]))))
    graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)
    graph = clip_branches(graph, img_bbox)
    graph, n_trimmed = prune_trim_branches(graph, prune_depth, trim_depth)

    # Extract irreducibles
    irreducibles = list()
    for node_subset in nx.connected_components(graph):
        if len(node_subset) + n_trimmed > min_size:
            subgraph = graph.subgraph(node_subset)
            irreducibles_i = get_subcomponent_irreducibles(
                subgraph, swc_dict, smooth_bool
            )
            if irreducibles_i:
                irreducibles.append(irreducibles_i)
    return irreducibles


def clip_branches(graph, img_bbox):
    """
    Deletes all nodes from "graph" that are not contained in "img_bbox".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched
    img_bbox : dict
        Dictionary with the keys "min" and "max" which specify a bounding box
        in the image. The default is None.

    Returns
    -------
    networkx.Graph
        "graph" with nodes deleted that were not contained in "img_bbox".

    """
    if img_bbox:
        delete_nodes = set()
        for i in graph.nodes:
            xyz = img_util.to_voxels(graph.nodes[i]["xyz"])
            if not util.is_contained(img_bbox, xyz):
                delete_nodes.add(i)
        graph.remove_nodes_from(delete_nodes)
    return graph


def prune_trim_branches(graph, prune_depth, trim_depth):
    """
    Prunes all short branches from "graph" and trims branchs if applicable. A
    short branch is a path between a leaf and junction node with a path length
    smaller than "prune_depth".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be pruned and trimmed.
    prune_depth : float
        Path length microns that determines whether a branch is short. The
        default is 16.0.
    trim_depth : float
        Maximum path length (in microns) to trim from "branch".

    Returns
    -------
    networkx.Graph
        Graph with branches trimmed and short branches pruned.

    """
    remove_nodes = list()
    n_nodes_trimmed = 0
    for leaf in get_leafs(graph):
        nodes, n_trimmed = prune_trim(graph, leaf, prune_depth, trim_depth)
        remove_nodes.extend(nodes)
        n_nodes_trimmed += n_trimmed
    graph.remove_nodes_from(remove_nodes)
    return graph, n_nodes_trimmed


def get_subcomponent_irreducibles(graph, swc_dict, smooth_bool):
    """
    Gets the irreducible components of "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    swc_dict : dict
        Dictionary used to build "graph".
    smooth : bool
        Indication of whether to smooth edges.

    Returns
    -------
    dict
        Dictionary containing irreducible components of "graph".

    """
    # Extract nodes
    leafs, junctions = get_irreducible_nodes(graph)
    assert len(leafs) > 0, "No leaf nodes!"
    if len(leafs) > 0:
        source = util.sample_once(leafs)
    else:
        source = util.sample_once(junctions)

    # Extract edges
    edges = dict()
    nbs = defaultdict(list)
    root = None
    for (i, j) in nx.dfs_edges(graph, source=source):
        # Check if starting new or continuing current path
        if root is None:
            root = i
            cur_length = 0
            attrs = init_edge_attrs(swc_dict, root)
        else:
            xyz_i = swc_dict["xyz"][swc_dict["idx"][i]]
            xyz_j = swc_dict["xyz"][swc_dict["idx"][j]]
            cur_length += geometry.dist(xyz_i, xyz_j)

        # Visit j
        attrs = upd_edge_attrs(swc_dict, attrs, j)
        if j in leafs or j in junctions:
            # Check whether to smooth
            attrs["length"] = cur_length
            attrs = to_numpy(attrs)
            if smooth_bool:
                swc_dict, edges = __smooth_branch(
                    swc_dict, attrs, edges, nbs, root, j
                )
            else:
                edges[(root, j)] = attrs

            # Finish
            nbs[root].append(j)
            nbs[j].append(root)
            root = None

    # Output
    swc_id = swc_dict["swc_id"]
    leafs = set_node_attrs(swc_dict, leafs)
    junctions = set_node_attrs(swc_dict, junctions)
    irreducibles = {
        "leafs": leafs,
        "junctions": junctions,
        "edges": edges,
        "swc_id": swc_id,
    }
    return irreducibles


def get_irreducible_nodes(graph):
    """
    Gets irreducible nodes (i.e. leafs and junctions) of a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    set, set
        Nodes with degreee 1 and degree > 2.

    """
    leafs = set()
    junctions = set()
    for i in graph.nodes:
        if graph.degree[i] == 1:
            leafs.add(i)
        elif graph.degree[i] > 2:
            junctions.add(i)
    return leafs, junctions


# --- Refine graph ---
def prune_trim(graph, leaf, prune_depth, trim_depth):
    """
    Determines whether the branch emanating from "leaf" should be pruned and
    returns nodes that should be pruned. If applicable (i.e. trim_depth > 0),
    every branch is trimmed by"trim_depth" microns.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    leaf : int
        Leaf node being inspected to determine whether it is the endpoint of
        a short branch that should be pruned.
    prune_depth : int
        Path length microns that determines whether a branch is short.
    trim_depth : float
        Maximum path length (in microns) to trim from "branch".

    Returns
    -------
    list
        If branch is short, then the list of nodes in the branch is returned.
        Otherwise, an empty list is returned.

    """
    # Check whether to prune
    branch = [leaf]
    node_spacing = list()
    for (i, j) in nx.dfs_edges(graph, source=leaf, depth_limit=prune_depth):
        node_spacing.append(compute_dist(graph, i, j))
        if graph.degree(j) > 2:
            return branch, 0
        elif graph.degree(j) == 2:
            branch.append(j)
        elif np.sum(node_spacing) > prune_depth:
            break

    # Check whether to trim
    spacing = np.mean(node_spacing)
    if trim_depth > 0 and graph.number_of_nodes() > 3 * trim_depth / spacing:
        trim_nodes = trim_branch(graph, branch, trim_depth)
        return trim_nodes, len(trim_nodes)
    else:
        return list(), 0


def trim_branch(graph, branch, trim_depth):
    """
    Trims a branch of a graph based on a specified depth.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    branch : list[int]
        List of nodes representing a path through the graph to be trimmed.
    trim_depth : float
        Maximum path length (in microns) to trim from "branch".

    Returns
    -------
    list[int]
        Trimmed branch.

    """
    i = 1
    path_length = 0
    for i in range(1, len(branch)):
        xyz_1 = graph.nodes[branch[i - 1]]["xyz"]
        xyz_2 = graph.nodes[branch[i]]["xyz"]
        path_length += geometry.dist(xyz_1, xyz_2)
        if path_length > trim_depth:
            break
    return branch[0:i]


def compute_dist(graph, i, j):
    """
    Computes Euclidean distance between nodes i and j.

    Parameters
    ----------
    graph : netowrkx.Graph
        Graph containing nodes i and j.
    i : int
        Node.
    j : int
        Node.

    Returns
    -------
    Euclidean distance between i and j.

    """
    return geometry.dist(graph.nodes[i]["xyz"], graph.nodes[j]["xyz"])


def __smooth_branch(swc_dict, attrs, edges, nbs, root, j):
    """
    Smoothes a branch then updates "swc_dict" and "edges" with the new xyz
    coordinates of the branch end points. Note that this branch is an edge
    in the irreducible graph being built.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of edge being smoothed.
    edges : dict
        Dictionary where the keys are edges in irreducible graph and values
        are the corresponding attributes.
    nbs : dict
        Dictionary where the keys are nodes and values are the neighbors.
    root : int
        End point of branch to be smoothed.
    j : int
        End point of branch to be smoothed.

    Returns
    -------
    dict, dict
        Dictionaries that have been updated with respect to smoothed edges.

    """
    attrs["xyz"] = geometry.smooth_branch(attrs["xyz"], s=2)
    swc_dict, edges = upd_xyz(swc_dict, attrs, edges, nbs, root, 0)
    swc_dict, edges = upd_xyz(swc_dict, attrs, edges, nbs, j, -1)
    edges[(root, j)] = attrs
    return swc_dict, edges


def upd_xyz(swc_dict, attrs, edges, nbs, i, endpoint):
    """
    Updates "swc_dict" and "edges" with the new xyz coordinates of the branch
    end points.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of edge being smoothed.
    edges : dict
        Dictionary where the keys are edges in irreducible graph and values
        are the corresponding attributes.
    nbs : dict
        Dictionary where the keys are nodes and values are the neighbors.
    endpoint : int
        End point of branch to be smoothed.

    Returns
    -------
    dict
        Updated with new xyz coordinates.
    dict
        Updated with new xyz coordinates.

    """
    idx = swc_dict["idx"][i]
    if i in nbs.keys():
        for j in nbs[i]:
            key = (i, j) if (i, j) in edges.keys() else (j, i)
            edges = upd_endpoint_xyz(
                edges, key, swc_dict["xyz"][idx], attrs["xyz"][endpoint]
            )
    swc_dict["xyz"][idx] = attrs["xyz"][endpoint]
    return swc_dict, edges


def upd_endpoint_xyz(edges, key, old_xyz, new_xyz):
    """
    Updates "edges" with the new xyz coordinates of the branch
    end points.

    Parameters
    ----------
    edges : dict
        Dictionary where the keys are edges in irreducible graph and values
        are the corresponding attributes.
    key : tuple
        The edge id of the entry in "edges" which needs to be updated.
    old_xyz : numpy.ndarray
        Current xyz coordinate of end point.
    new_xyz : numpy.ndarray
        New xyz coordinate of end point.

    Returns
    -------
    dict
        Updated with new xyz coordinates.

    """
    if all(edges[key]["xyz"][0] == old_xyz):
        edges[key]["xyz"][0] = new_xyz
    elif all(edges[key]["xyz"][-1] == old_xyz):
        edges[key]["xyz"][-1] = new_xyz
    return edges


# -- attribute utils --
def init_edge_attrs(swc_dict, i):
    """
    Initializes edge attribute dictionary with attributes from node "i" which
    is an end point of the edge. Note: the following assertion error may be
    useful: assert i in swc_dict["idx"].keys(), f"{swc_dict["swc_id"]} - {i}"

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    i : int
        End point of edge and the swc attributes of this node are used to
        initialize the edge attriubte dictionary.

    Returns
    -------
    dict
        Edge attribute dictionary.

    """
    j = swc_dict["idx"][i]
    return {"radius": [swc_dict["radius"][j]], "xyz": [swc_dict["xyz"][j]]}


def upd_edge_attrs(swc_dict, attrs, i):
    """
    Updates an edge attribute dictionary with attributes of node i.

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    attrs : dict
        Attributes (from "swc_dict") of edge being updated.
    i : int
        Node of edge whose attributes will be added to "attrs".

    Returns
    -------
    dict
        Edge attribute dictionary.

    """
    swc_id = swc_dict["swc_id"]
    assert i != -1, f"{swc_id} - {i}"
    j = swc_dict["idx"][i]
    attrs["radius"].append(swc_dict["radius"][j])
    attrs["xyz"].append(swc_dict["xyz"][j])
    return attrs


def get_edge_attr(graph, edge, attr):
    """
    Gets the attribute "attr" of "edge".

    Parameters
    ----------
    graph : networkx.Graph
        Graph which "edge" belongs to.
    edge : tuple
        Edge to be queried for its attributes.
    attr : str
        Attribute to be queried.

    Returns
    -------
    Attribute "attr" of "edge"

    """
    return graph.edges[edge][attr]


def to_numpy(attrs):
    """
    Converts edge attributes from a list to NumPy array.

    Parameters
    ----------
    attrs : dict
        Dictionary containing attributes of some edge.

    Returns
    -------
    dict
        Updated edge attribute dictionary.

    """
    attrs["xyz"] = np.array(attrs["xyz"], dtype=np.float32)
    attrs["radius"] = np.array(attrs["radius"], dtype=np.float16)
    return attrs


def set_node_attrs(swc_dict, nodes):
    """
    Set node attributes by extracting values from "swc_dict".

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file.
    nodes : list
        List of nodes to set attributes.

    Returns
    -------
    dict
        Dictionary in which keys are node ids and values are a dictionary of
        attributes extracted from "swc_dict".

    """
    attrs = dict()
    for i in nodes:
        j = swc_dict["idx"][i]
        attrs[i] = {"radius": swc_dict["radius"][j], "xyz": swc_dict["xyz"][j]}
    return attrs


def upd_node_attrs(swc_dict, leafs, junctions, i):
    """
    Updates node attributes by extracting values from "swc_dict".

    Parameters
    ----------
    swc_dict : dict
        Contents of an swc file that contains the smoothed xyz coordinates of
        corresponding to "leafs" and "junctions". Note xyz coordinates are
        smoothed during edge extraction.
    leafs : dict
        Dictionary where keys are leaf node ids and values are attribute
        dictionaries.
    junctions : dict
        Dictionary where keys are junction node ids and values are attribute
        dictionaries.
    i : int
        Node to be updated.

    Returns
    -------
    dict
        Updated dictionary if "i" was contained in "leafs.keys()".
    dict
        Updated dictionary if "i" was contained in "junctions.keys()".

    """
    j = swc_dict["idx"][i]
    upd_attrs = {"radius": swc_dict["radius"][j], "xyz": swc_dict["xyz"][j]}
    if i in leafs:
        leafs[i] = upd_attrs
    else:
        junctions[i] = upd_attrs
    return leafs, junctions


# -- miscellaneous --
def creates_cycle(graph, edge):
    """
    Checks whether adding "edge" to "graph" creates a cycle.

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be checked for cycles.
    edge : tuple
        Edge to be added to "graph"

    Returns
    -------
    bool
        Indication of whether adding "edge" to graph creates a cycle.

    """
    graph.add_edges_from([edge])
    exists = cycle_exists(graph)
    graph.remove_edges_from([edge])
    if exists:
        return True, edge
    else:
        return False, edge


def cycle_exists(graph):
    """
    Checks whether a cycle exists in "graph".

    Paramaters
    ----------
    graph : networkx.Graph
        Graph to be checked for cycles.

    Returns
    -------
    bool
        Indication of whether there exists a cycle.

    """
    try:
        nx.find_cycle(graph)
        return True
    except nx.exception.NetworkXNoCycle:
        return False


def get_leafs(graph):
    """
    Gets leaf nodes of "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched

    Returns
    -------
    list
        Leaf nodes "graph".
    """
    return [i for i in graph.nodes if graph.degree[i] == 1]


def sample_node(graph):
    """
    Samples a single node from a graph.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be sampled from.

    Returns
    -------
    int
        Node.

    """
    nodes = list(graph.nodes)
    return sample(nodes, 1)[0]


def get_component(graph, root):
    """
    Gets the connected component corresponding to "root" from "graph".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    root : int
        Node that breadth-first search starts from.

    Returns
    -------
    set[int]
        Set of nodes in the connected component corresponding to "root".

    """
    queue = [root]
    component = set()
    while len(queue):
        i = queue.pop()
        component.add(i)
        for j in [j for j in graph.neighbors(i) if j not in component]:
            if (i, j) in graph.edges:
                queue.append(j)
    return component


def count_components(graph):
    """
    Counts the number of connected components in a graph.

    Paramters
    ---------
    graph : networkx.Graph
        Graph to be searched.

    Returns
    -------
    Number of connected components.

    """
    return nx.number_connected_components(graph)


def largest_components(neurograph, k):
    """
    Finds the "k" largest connected components in "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be searched.
    k : int
        Number of largest connected components to return.

    Returns
    -------
    list
        List where each entry is a random node from one of the k largest
        connected components.

    """
    component_cardinalities = k * [-1]
    node_ids = k * [-1]
    for nodes in nx.connected_components(neurograph):
        if len(nodes) > component_cardinalities[-1]:
            i = 0
            while i < k:
                if len(nodes) > component_cardinalities[i]:
                    component_cardinalities.insert(i, len(nodes))
                    component_cardinalities.pop(-1)
                    node_ids.insert(i, util.sample_singleton(nodes))
                    node_ids.pop(-1)
                    break
                i += 1
    return node_ids
