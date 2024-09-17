"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds a neurograph for neuron reconstruction.

"""

from time import time

from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs.utils import graph_util as gutil
from deep_neurographs.utils import img_util, swc_util

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
        irreducibles = gutil.get_irreducibles(
            swc_dicts,
            self.min_size,
            self.img_bbox,
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
