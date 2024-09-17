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
    Represents configuration settings related to graph properties.

    Attributes
    ----------
    anisotropy : list[float], optional
        Scaling factors applied to xyz coordinates to account for anisotropy
        of microscope. The default is [1.0, 1.0, 1.0].
    img_bbox : dict, optional
        Dictionary with the keys "min" and "max" which specify a bounding box
        in an image. The default is None.
    min_size : float, optional
        Minimum path length (in microns) of swc files which are stored as
        connected components in the FragmentsGraph. The default is 30um.
    node_spacing : int, optional
        Spacing (in microns) between nodes. The default is 1.
    prune_depth : int, optional
        Branches in graph less than "prune_depth" microns are pruned. The
        default is 16.
    smooth_bool : bool, optional
        Indication of whether to smooth branches in graph. The default is
        True.
    trim_depth : float, optional
        Maximum path length (in microns) to trim from all branches. The
        default is 5.

    """
    anisotropy: List[float] = field(default_factory=list)
    img_bbox: dict = None
    min_size: float = 30.0
    node_spacing: int = 1
    prune_depth: float = 16.0
    remove_doubles_bool: bool = False
    smooth_bool: bool = True
    trim_depth: float = 5.0


@dataclass
class ProposalsConfig:
    """
    Represents configuration settings related to proposals generated.

    Attributes
    ----------
    ...

    """
    complex_bool: bool = False
    long_range_bool: bool = True
    proposals_per_leaf: int = 2
    search_radius: float = 20.0
    trim_endpoints_bool: bool = True


@dataclass
class MLConfig:
    batch_size: int = 1600
    downsample_factor: int = 1
    high_threshold: float = 0.9
    threshold: float = 0.6
    model_type: str = "GraphNeuralNet"


class Config:
    def __init__(
        self,
        graph_config: GraphConfig,
        proposals_config: ProposalsConfig,
        ml_config: MLConfig
    ):
        self.graph_config = graph_config
        self.proposals_config = proposals_config
        self.ml_config = ml_config
