"""
Created on Sat Sept 16 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This module defines a set of configuration classes used for setting up various
aspects of a system involving graphs, proposals, and machine learning (ML).

"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class GraphConfig:
    """
    Represents configuration settings related to graph properties and
    proposals generated.

    Attributes
    ----------
    anisotropy : list[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope. Note this instance of "anisotropy" is only used while
        reading fragments (i.e. swcs). The default is [1.0, 1.0, 1.0].
    complex_bool : bool
        Indication of whether to generate complex proposals, meaning proposals
        between leaf and non-leaf nodes. The default is False.
    img_bbox : dict, optional
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image. The default is None.
    long_range_bool : bool, optional
        Indication of whether to generate simple proposals within a scaled
        distance of "search_radius" from leaves without any proposals. The
        default is False.
    min_size : float, optional
        Minimum path length (in microns) of swc files which are stored as
        connected components in the FragmentsGraph. The default is 30.
    node_spacing : int, optional
        Spacing (in microns) between nodes. The default is 2.
    proposals_per_leaf : int
        Maximum number of proposals generated for each leaf. The default is 3.
    prune_depth : int, optional
        Branches in graph less than "prune_depth" microns are pruned. The
        default is 16.
    smooth_bool : bool, optional
        Indication of whether to smooth branches in the graph. The default is
        True.
    trim_endpoints_bool : bool, optional
        Indication of whether to endpoints of branches with exactly one
        proposal. The default is True.

    """

    anisotropy: List[float] = field(default_factory=list)
    complex_bool: bool = False
    img_bbox: dict = None
    long_range_bool: bool = True
    min_size: float = 30.0
    node_spacing: int = 2
    proposals_per_leaf: int = 2
    prune_depth: float = 16.0
    remove_doubles_bool: bool = False
    search_radius: float = 20.0
    smooth_bool: bool = True
    trim_endpoints_bool: bool = True


@dataclass
class MLConfig:
    """
    Configuration class for machine learning model parameters.

    Attributes
    ----------
    anisotropy : list[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope. Note this instance of "anisotropy" is only used while
        generating features. The default is [1.0, 1.0, 1.0].
    batch_size : int
        The number of samples processed in one batch during training or
        inference. Default is 1000.
    multiscale : int
        Level in the image pyramid that voxel coordinates must index into.
    high_threshold : float
        A threshold value used for classification, above which predictions are
        considered to be high-confidence. Default is 0.9.
    threshold : float
        A general threshold value used for classification. Default is 0.6.
    model_type : str
        Type of machine learning model to use. Default is "GraphNeuralNet".

    """
    anisotropy: List[float] = field(default_factory=list)
    batch_size: int = 2000
    high_threshold: float = 0.9
    lr: float = 1e-3
    model_type: str = "GraphNeuralNet"
    multiscale: int = 1
    n_epochs: int = 1000
    threshold: float = 0.6
    validation_split: float = 0.15
    weight_decay: float = 1e-3


class Config:
    """
    A configuration class for managing and storing settings related to graph
    and machine learning models.

    """

    def __init__(self, graph_config: GraphConfig, ml_config: MLConfig):
        """
        Initializes a Config object which is used to manage settings used to
        run a GraphTrace pipeline.

        Parameters
        ----------
        graph_config : GraphConfig
            Instance of the "GraphConfig" class that contains configuration
            parameters for graph and proposal operations, such as anisotropy,
            node spacing, and other graph-specific settings.
        ml_config : MLConfig
            An instance of the "MLConfig" class that includes configuration
            parameters for machine learning models.

        """
        self.graph_config = graph_config
        self.ml_config = ml_config
