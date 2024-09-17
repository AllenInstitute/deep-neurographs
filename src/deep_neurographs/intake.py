"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds a neurograph for neuron reconstruction.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
from tqdm import tqdm

import numpy as np

from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import img_util, swc_util, util

MIN_SIZE = 30
NODE_SPACING = 1
SMOOTH_BOOL = True
PRUNE_DEPTH = 16
TRIM_DEPTH = 0


class GraphBuilder:
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
            Indication of whether to print out a progress bar during build.
            The default is False.
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
        t0 = time()
        self.set_img_bbox(img_patch_origin, img_patch_shape)
        swc_dicts = self.reader.load(fragments_pointer)
        irreducibles = self.get_irreducibles(swc_dicts)

        # Build FragmentsGraph
        neurograph = NeuroGraph(node_spacing=self.node_spacing)
        while len(irreducibles):
            irreducible_set = irreducibles.pop()
            neurograph.add_component(irreducible_set)

        # Report Memory and runtime
        if self.progress_bar:
            usage = round(util.get_memory_usage(), 2)
            t, unit = util.time_writer(time() - t0)
            print(f"Memory Consumption: {usage} GBs")
            print(f"Module Runtime: {round(t, 4)} {unit} \n")
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

    def get_irreducibles(self, swc_dicts):
        """
        Gets irreducible components of each graph stored in "swc_dicts" by
        setting up a parellelization scheme that sends each graph to a CPU and
        calling the subroutine "gutil.get_irreducibles".

        Parameters
        ----------
        swc_dicts : list[dict]
            List of dictionaries such that each entry contains the conents of
            an swc file.

        Returns
        -------
        list[dict]
            List of irreducibles stored in a dictionary where key-values are
            type of irreducible (i.e. leaf, junction, or edge) and the
            corresponding set of all irreducibles from the graph of that type.

        """
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
            with tqdm(total=len(processes), desc="Extract Graphs") as pbar:
                irreducibles = []
                n_nodes, n_edges = 0, 0
                for process in as_completed(processes):
                    irreducibles_i = process.result()
                    irreducibles.extend(irreducibles_i)
                    n_nodes += count_nodes(irreducibles_i)
                    n_edges += count_edges(irreducibles_i)
                    pbar.update(1)

        # Report graph size
        if self.progress_bar:
            n_components = util.reformat_number(len(irreducibles))
            print("\nGraph Overview...")
            print("# connected components:", n_components)
            print("# nodes:", util.reformat_number(n_nodes))
            print("# edges:", util.reformat_number(n_edges))
        return irreducibles


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
    return np.sum([len(irr["edges"]) for irr in irreducibles])
