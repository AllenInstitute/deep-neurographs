"""
Created on Mon March 6 19:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import networkx as nx


def prune_spurious_paths(graph, min_branch_length=16):
    """
    Prunes short branches.

    Parameters
    ----------
    graph : networkx.graph
        Graph to be pruned.
    min_branch_length : int, optional
        Upper bound on short branch length to be pruned. The default is 16.

    Returns
    -------
    graph : networkx.Graph
        Graph with short branches pruned.

    """
    leaf_nodes = [i for i in graph.nodes if graph.degree[i] == 1]
    for leaf in leaf_nodes:
        # Traverse branch from leaf
        queue = [leaf]
        visited = set()
        hit_junction = False
        while len(queue) > 0:
            node = queue.pop(0)
            nbs = list(graph.neighbors(node))
            if len(nbs) > 2:
                hit_junction = True
                break
            else:
                visited.add(node)
                nb = [nb for nb in nbs if nb not in visited]
                queue.extend(nb)

        # Check length of branch
        if hit_junction and len(visited) <= min_branch_length:
            graph.remove_nodes_from(visited)
    return graph


def prune_short_connectors(graph, connector_dist=8):
    """ "
    Prunes shorts paths (i.e. connectors) between junctions nodes and the nbhd about the
    junctions.

    Parameters
    ----------
    graph : netowrkx.graph
        Graph to be inspected.
    connector_dist : int
        Upper bound on the distance that defines a connector path to be pruned.

    Returns
    -------
    graph : list[tuple]
        Graph with connectors pruned
    pruned_centroids : list[np.ndarray]
        List of xyz coordinates of centroids of connectors

    """
    junctions = [j for j in graph.nodes if graph.degree[j] > 2]
    pruned_centroids = []
    pruned_nodes = set()
    while len(junctions):
        # Search nbhd
        j = junctions.pop()
        junction_nbs = []
        for _, i in nx.dfs_edges(graph, source=j, depth_limit=connector_dist):
            if graph.degree[i] > 2 and i != j:
                junction_nbs.append(i)

        # Store nodes to be pruned
        print("# junction nbs:", len(junction_nbs))
        for nb in junction_nbs:
            connector = list(nx.shortest_path(graph, source=j, target=nb))
            nbhd = set(nx.dfs_tree(graph, source=nb, depth_limit=5))
            centroid = connector[len(connector) // 2]
            pruned_nodes.update(nbhd.union(set(connector)))
            pruned_centroids.append(graph.nodes[centroid]["xyz"])

        if len(junction_nbs) > 0:
            nbhd = set(nx.dfs_tree(graph, source=j, depth_limit=8))
            pruned_nodes.update(nbhd)
        break

    graph.remove_nodes_from(list(pruned_nodes))
    return graph, pruned_centroids


def break_crossovers(list_of_graphs, depth=10):
    """
    Breaks crossovers for each graph contained in "list_of_graphs".

    Parameters
    ----------
    list_of_graphs : list[networkx.graph]
        List of graphs such that crossovers will be broken on each graph.
    depth : int, optional
        Maximum depth of dfs performed to detect crossovers. The default is 10.
    prune : bool, optional
        Indicates whether to prune spurious branches. The default is True.

    Returns
    -------
    upd : list[networkx.graph]
        List of graphs with crossovers broken.

    """
    upd = []
    for i, graph in enumerate(list_of_graphs):
        pruned_graph = prune_spurious_paths(graph, min_branch_length=depth)
        prune_nodes = detect_crossovers(pruned_graph, depth)
        if len(prune_nodes) > 0:
            graph.remove_nodes_from(prune_nodes)
            for g in nx.connected_components(graph):
                subgraph = graph.subgraph(g).copy()
                if subgraph.number_of_nodes() > 10:
                    upd.append(subgraph)
        else:
            upd.append(graph)
    return upd


def detect_crossovers(graph, depth):
    """
    Detects crossovers in "graph".

    Parameters
    ----------
    graph : networkx.graph
        Graph to be inspected.
    depth : int
        Maximum depth of dfs performed to detect crossovers.

    Returns
    -------
    prune_nodes : list[int]
        Nodes that are part of a crossover and should be pruned.

    """
    cnt = 0
    prune_nodes = []
    junctions = [j for j in graph.nodes if graph.degree(j) > 2]
    for j in junctions:
        # Explore node
        upd = False
        tree, leafs = count_branches(graph, j, depth)
        num_leafs = len(leafs)

        # Detect crossover
        if num_leafs > 3:
            cnt += 1
            upd = True
            for d in range(1, depth):
                tree_d, leafs_d = count_branches(graph, j, d)
                if len(leafs_d) == num_leafs:
                    prune_nodes.extend(tree_d.nodes())
                    upd = False
                    break
            if upd:
                prune_nodes.extend(tree.nodes())
    return prune_nodes


def count_branches(graph, source, depth):
    """
    Counts the number of branches emanating from "source" by running a
    bounded dfs.

    Parameters
    ----------
    graph : networkx.graph
        Graph that contains "source".
    source : int
        Node that is contained in "graph".
    depth : int
        Maximum depth of dfs.

    Returns
    -------
    tree : networkx.dfs_tree
        Tree-structured graph rooted at "source".
    leafs : list[int]
        List of leaf nodes in "tree".

    """
    tree = nx.dfs_tree(graph, source=source, depth_limit=depth)
    leafs = [i for i in tree.nodes() if tree.degree(i) == 1]
    return tree, leafs
