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


class SkeletonGraph(nx.Graph):

    def __init__(self):
        # Call parent class
        super().__init__()

    def load_skeletons(self, swc_pointer):
        pass
