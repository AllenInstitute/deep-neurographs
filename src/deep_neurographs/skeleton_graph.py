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

from deep_neurographs.utils import graph_util as gutil

class SkeletonGraph(nx.Graph):

    def __init__(
        anisotropy=(1.0, 1.0, 1.0),
        min_size=0,
        node_spacing=1,
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
            verbose=verbose,
        )

        # Instance attributes

    def load_skeletons(self, swc_pointer):
        # Extract irreducible components from SWC files
        irreducibles = self.graph_loader.run(swc_pointer)
        n_irreducible = np.sum([len(irr["nodes"]) for irr in irreducibles])
        n_reducible = np.sum([len(irr["edges"]) - 2 for irr in irreducibles])
        n = n_irreducible + n_reducible

        # Initialize node attribute data structures
        self.node_component_id = np.zeros((n), dtype=int)
        self.node_radius = np.zeros((n), dtype=np.float16)
        self.node_xyz = np.zeros((n, 3), dtype=np.float32)

        # Add irreducibles to graph
        component_id = 0
        while irreducibles:
            self.add_connected_component(irreducibles.pop(), component_id)
            component_id += 1
