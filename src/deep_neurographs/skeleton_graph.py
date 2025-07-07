"""
Created on Wed July 2 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of a custom subclass of Networkx.Graph called "SkeletonGraph".
After initializing an instance of this subclass, the graph is built by reading
and processing SWC files (i.e. neuron fragments). It then stores the relevant
information into the graph structure.

"""

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

        # Skeleton Loader
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
        self.irreducible.add_nodes_from(list(node_id_mapping.values()))

        # Add edges
        for (i, j), attrs in irreducibles["edges"].items():
            edge_id = (node_id_mapping[i], node_id_mapping[j])
            self._add_edge(edge_id, attrs, node_id_mapping)
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
        iterator = zip(attrs["radius"][1:-1], attrs["xyz"][1:-1])
        for cnt, (radius, xyz) in enumerate(iterator):
            # Add edge
            new_id = self.number_of_nodes()
            if cnt == 0:
                self.add_edge(start, new_id)
            elif cnt == len(attrs["xyz"]) - 2:
                self.add_edge(new_id, end)
            else:
                self.add_edge(new_id, new_id - 1)

            # Store attributes
            self.node_xyz[new_id] = xyz
            self.node_radius[new_id] = radius
