"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of a custom subclass of Networkx.Graph called "SkeletonGraph".
After initializing an instance of this subclass, the graph is built by reading
and processing SWC files (i.e. neuron fragments). It then stores the relevant
information into the graph structure.

"""

from scipy.spatial import KDTree

import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, graph_util as gutil


class SkeletonGraph(nx.Graph):

    def __init__(
        self,
        anisotropy=(1.0, 1.0, 1.0),
        min_size=0,
        node_spacing=1,
        prune_depth=20.0,
        smooth_bool=True,
        verbose=False,
    ):
        # Call parent class
        super().__init__()

        # Graph Loader
        self.graph_loader = gutil.GraphLoader(
            anisotropy=anisotropy,
            min_size=min_size,
            node_spacing=node_spacing,
            smooth_bool=smooth_bool,
            prune_depth=prune_depth,
            verbose=verbose,
        )

        # Instance attributes
        self.component_id_to_swc_id = dict()
        self.irreducible = nx.Graph()

    # --- Build Graph ---
    def load(self, swc_pointer):
        # Extract irreducible components from SWC files
        irreducibles = self.graph_loader.run(swc_pointer)
        n = 0
        for irr in irreducibles:
            n += len(irr["nodes"])
            for attrs in irr["edges"].values():
                n += len(attrs["xyz"]) - 2

        # Initialize node attribute data structures
        self.node_component_id = np.zeros((n), dtype=int)
        self.node_radius = np.zeros((n), dtype=np.float16)
        self.node_xyz = np.zeros((n, 3), dtype=np.float32)

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1

    def add_connected_component(self, irreducibles, component_id):
        # Set component id
        self.component_id_to_swc_id[component_id] = irreducibles["swc_id"]

        # Add nodes
        node_id_mapping = self._add_nodes(irreducibles["nodes"], component_id)

        # Add edges
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs, component_id)
            self.irreducible.add_edge(*edge_id)

    def _add_nodes(self, node_dict, component_id):
        """
        Adds nodes to the graph from a dictionary of node attributes and
        returns a mapping from original node IDs to the new graph node IDs.

        Parameters
        ----------
        node_dict : dict
            Dictionary mapping original node IDs (e.g., from an SWC file) to
            their attributes. Each value must be a dictionary containing:
                - "radius" : float
                - "xyz"    : array-like of shape (3,)
        component_id : str
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".

        Returns
        -------
        dict
            Dictionary mapping the original node IDs from "node_dict" to the
            new node IDs assigned in the graph.
        """
        node_id_mapping = dict()
        for node_id, attrs in node_dict.items():
            new_id = self.number_of_nodes()
            self.node_xyz[new_id] = attrs["xyz"]
            self.node_radius[new_id] = attrs["radius"]
            self.node_component_id[new_id] = component_id
            self.add_node(new_id)
            node_id_mapping[node_id] = new_id
        return node_id_mapping

    def _add_edge(self, edge_id, attrs, component_id):
        """
        Adds an edge to "self".

        Parameters
        ----------
        edge : Tuple[int]
            Edge to be added.
        attrs : dict
            Dictionary of attributes of "edge" obtained from an SWC file.
        component_id : str
            Connected component ID used to map node IDs back to SWC IDs via
            "self.component_id_to_swc_id".

        Returns
        -------
        None
        """
        # Determine orientation of attributes
        i, j = tuple(edge_id)
        dist_i = geometry_util.dist(self.node_xyz[i], attrs["xyz"][0])
        dist_j = geometry_util.dist(self.node_xyz[j], attrs["xyz"][0])
        if dist_i < dist_j:
            start = i
            end = j
        else:
            start = j
            end = i

        # Populate graph
        iterator = zip(attrs["radius"], attrs["xyz"])
        for cnt, (radius, xyz) in enumerate(iterator):
            if cnt > 0 and cnt < len(attrs["xyz"]) - 1:
                # Add edge
                new_id = self.number_of_nodes()
                if cnt == 1:
                    self.add_edge(start, new_id)
                else:
                    self.add_edge(new_id, new_id - 1)

                # Store attributes
                self.node_xyz[new_id] = xyz
                self.node_radius[new_id] = radius
                self.node_component_id[new_id] = component_id
        self.add_edge(new_id, end)

    def relabel_nodes(self):
        # Set node ids
        old_node_ids = np.array(self.nodes, dtype=int)
        new_node_ids = np.arange(len(old_node_ids))

        # Set edge ids
        old_to_new = dict(zip(old_node_ids, new_node_ids))
        old_edge_ids = list(self.edges)
        old_irr_edge_ids = self.irreducible.edges

        # Reset graph
        self.clear()
        for (i, j) in old_edge_ids:
            self.add_edge(old_to_new[i], old_to_new[j])

        self.irreducible.clear()
        for (i, j) in old_irr_edge_ids:
            self.irreducible.add_edge(old_to_new[i], old_to_new[j])

        self.node_xyz = self.node_xyz[old_node_ids]
        self.node_component_id = self.node_component_id[old_node_ids]

    # --- Getters ---
    def get_leafs(self):
        """
        Gets all leaf nodes in graph.

        Parameters
        ----------
        None

        Returns
        -------
        List[int]
            Leaf nodes in graph.
        """
        return [i for i in self.nodes if self.degree[i] == 1]

    def get_nodes_with_component_id(self, component_id):
        return set(np.where(self.node_component_id == component_id)[0])

    def get_rooted_subgraph(self, root, radius):
        # Initializations
        subgraph = SkeletonGraph()
        node_mapping = {root: 0}

        # Extract graph
        queue = [(root, 0)]
        visited = {root}
        while queue:
            # Visit node
            i, dist_i = queue.pop()

            # Populate queue
            for j in self.neighbors(i):
                dist_j = dist_i + self.dist(i, j)
                if j not in visited and dist_j < radius:
                    node_mapping[j] = subgraph.number_of_nodes()
                    subgraph.add_edge(node_mapping[i], node_mapping[j])
                    queue.append((j, dist_j))
                    visited.add(j)

        # Store coordinates
        idxs = np.array(list(node_mapping.keys()), dtype=int)
        subgraph.node_xyz = self.node_xyz[idxs]
        return subgraph

    def get_swc_id(self, i):
        component_id = self.node_component_id[i]
        return self.component_id_to_swc_id[component_id]

    def get_swc_ids(self):
        return np.unique(list(self.component_id_to_swc_id.values()))

    # --- Helpers ---
    def dist(self, i, j):
        """
        Computes the Euclidean distance between nodes "i" and "j".

        Parameters
        ----------
        i : int
            Node ID.
        j : int
            Node ID.

        Returns
        -------
        float
            Euclidean distance between nodes "i" and "j".
        """
        return geometry_util.dist(self.node_xyz[i], self.node_xyz[j])

    def init_kdtree(self):
        self.kdtree = KDTree(self.node_xyz)

    def query_node(self, xyz):
        # this breaks if node was deleted after kdtree was built
        _, idx = self.kdtree.query(xyz)
        return idx
