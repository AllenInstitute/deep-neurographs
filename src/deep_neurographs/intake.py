"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import os

from deep_neurographs import swc_utils, utils
from deep_neurographs.neurograph import NeuroGraph


# --- Build graph ---
def build_neurograph(
    swc_dir,
    anisotropy=[1.0, 1.0, 1.0],
    img_path=None,
    size_threshold=40,
    num_proposals=3,
    search_radius=25.0,
    prune=True,
    prune_depth=16,
    optimize_depth=15,
    optimize_alignment=True,
    optimize_path=False,
    origin=None,
    shape=None,
    smooth=True,
):
    """
    Builds a neurograph from a directory of swc files, where each swc
    represents a neuron and these neurons are assumed to be near each
    other.

    """
    neurograph = NeuroGraph(
        swc_dir,
        img_path=img_path,
        optimize_depth=optimize_depth,
        optimize_alignment=optimize_alignment,
        optimize_path=optimize_path,
        origin=origin,
        shape=shape,
    )
    neurograph = init_immutables(
        neurograph,
        anisotropy=anisotropy,
        prune=prune,
        prune_depth=prune_depth,
        size_threshold=size_threshold,
        smooth=smooth,
    )
    if search_radius > 0:
        neurograph.generate_proposals(
            num_proposals=num_proposals, search_radius=search_radius
        )
    return neurograph


def init_immutables(
    neurograph,
    anisotropy=[1.0, 1.0, 1.0],
    prune=True,
    prune_depth=16,
    size_threshold=40,
    smooth=True,
):
    """
    To do...
    """

    for path in get_paths(neurograph.path):
        swc_id = get_id(path)
        swc_dict = swc_utils.parse(
            path,
            anisotropy=anisotropy,
            bbox=neurograph.bbox,
            img_shape=neurograph.shape,
        )
        if len(swc_dict["xyz"]) < size_threshold:
            continue
        if smooth:
            swc_dict = swc_utils.smooth(swc_dict)
        neurograph.generate_immutables(
            swc_id, swc_dict, prune=prune, prune_depth=prune_depth
        )
    return neurograph


def get_paths(path_or_list):
    if type(path_or_list) == str:
        paths = []
        for f in utils.listdir(path_or_list, ext=".swc"):
            paths.append(os.path.join(path_or_list, f))
        return paths
    elif type(path_or_list) == list:
        return path_or_list


def get_id(path):
    filename = path.split("/")[-1]
    return filename.replace(".0.swc", "")
