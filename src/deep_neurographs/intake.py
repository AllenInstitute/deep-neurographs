"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds a neurograph for neuron reconstruction.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
from tqdm import tqdm

from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import img_util, swc_util, util

MIN_SIZE = 30
NODE_SPACING = 2
SMOOTH_BOOL = True
PRUNE_DEPTH = 25
TRIM_DEPTH = 0


class GraphBuilder:
    """
    Class that is used to build an instance of FragmentsGraph.

    """
    def __init__(
        self,
        anisotropy=[1.0, 1.0, 1.0],
        img_patch_origin=None,
        img_patch_shape=None,
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
        image_patch_origin : list[float], optional
            An xyz coordinate which is the upper, left, front corner of the
            image patch that contains the swc files. The default is None.
        image_patch_shape : list[float], optional
            Shape of the image patch which contains the swc files. The default
            is None.
        min_size : int, optional
            Minimum cardinality of swc files that are stored in NeuroGraph. The
            default is the global variable "MIN_SIZE".
        node_spacing : int, optional
            Spacing (in microns) between nodes. The default is the global
            variable "NODE_SPACING".
        progress_bar : bool, optional
            Indication of whether to print out a progress bar during build.
            The default is False.
        prune_depth : int, optional
            Branches less than "prune_depth" microns are pruned if "prune" is
            True. The default is the global variable "PRUNE_DEPTH".
        smooth : bool, optional
            Indication of whether to smooth branches from swc files. The
            default is the global variable "SMOOTH".
        trim_depth : float, optional
            Maximum path length (in microns) to trim from "branch". The default
            is the global variable "TRIM_DEPTH".

        Returns
        -------
        NeuroGraph
            Neurograph generated from swc files.

        """
        self.anisotropy = anisotropy
        self.min_size = min_size
        self.node_spacing = node_spacing
        self.progress_bar = progress_bar
        self.prune_depth = prune_depth
        self.smooth_bool = smooth_bool
        self.trim_depth = trim_depth

        self.img_bbox = img_util.get_bbox(img_patch_origin, img_patch_shape)
        self.reader = swc_util.Reader(anisotropy, min_size)

    def run(self, swc_pointer):
        """
        Builds a FragmentsGraph by reading swc files stored either on the
        cloud or local machine, then extracting the irreducible components.

        Parameters
        ----------
        swc_pointer : dict, list, str
            Pointer to swc files used to build an instance of FragmentsGraph,
            see "swc_util.Reader" for further documentation.

        Returns
        -------
        NeuroGraph
            Neurograph generated from swc files.

        """
        # Initializations
        t0 = time()
        swc_dicts = self.reader.load(swc_pointer)
        irreducibles, n_nodes, n_edges = self.get_irreducibles(swc_dicts)

        # Build FragmentsGraph
        neurograph = NeuroGraph(node_spacing=self.node_spacing)
        while len(irreducibles):
            irreducible_set = irreducibles.pop()
            neurograph.add_component(irreducible_set)

        # Report results
        if self.progress_bar:
            # Graph size
            n_components = util.reformat_number(len(irreducibles))
            print("\nGraph Overview...")
            print("# connected components:", n_components)
            print("# nodes:", util.reformat_number(n_nodes))
            print("# edges:", util.reformat_number(n_edges))

            # Memory and runtime
            usage = round(util.get_memory_usage(), 2)
            t, unit = util.time_writer(time() - t0)
            print(f"Memory Consumption: {usage} GBs")
            print(f"Module Runtime: {round(t, 4)} {unit} \n")
        return neurograph

    def get_irreducibles(self, swc_dicts):
        with ProcessPoolExecutor() as executor:
            # Assign Processes
            i = 0
            processes = [None] * len(swc_dicts)
            while swc_dicts:
                swc_dict = swc_dicts.pop()
                processes[i] = executor.submit(
                    gutil.get_irreducibles,
                    swc_dict,
                    self.min_size,
                    self.img_bbox,
                    self.prune_depth,
                    self.smooth_bool,
                    self.trim_depth,
                )
                i += 1

            # Store results
            desc = "Extract Graphs"
            irreducibles = []
            n_nodes, n_edges = 0, 0
            for process in tqdm(as_completed(processes), desc=desc):
                irreducibles_i = process.result()
                irreducibles.extend(irreducibles_i)
                n_nodes += count_nodes(irreducibles_i)
                n_edges += count_edges(irreducibles_i)
        return irreducibles, n_nodes, n_edges


# --- utils ---
def count_nodes(irreducibles):
    """
    Counts the number of nodes in "irreducibles".

    Parameters
    ----------
    irreducibles : dict
        Dictionary that contains the irreducible components of a graph.

    Returns
    -------
    int
        Number of nodes in "irreducibles".

    """
    cnt = 0
    for irr_i in irreducibles:
        cnt += len(irr_i["leafs"]) + len(irr_i["junctions"])
    return cnt


def count_edges(irreducibles):
    """
    Counts the number of edges in "irreducibles".

    Parameters
    ----------
    irreducibles : dict
        Dictionary that contains the irreducible components of a graph.

    Returns
    -------
    int
        Number of edges in "irreducibles".

    """
    cnt = 0
    for irr_i in irreducibles:
        cnt += len(irr_i["edges"])
    return cnt
