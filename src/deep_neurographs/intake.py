"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import os

from deep_neurographs import s3_utils, swc_utils, utils
from deep_neurographs.neurograph import NeuroGraph


# --- Build graph ---
def build_neurograph(
    swc_dir,
    anisotropy=[1.0, 1.0, 1.0],
    img_path=None,
    num_proposals=3,
    search_radius=25.0,
    prune=True,
    prune_depth=16,
    optimize_proposals=False,
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
        optimize_proposals=optimize_proposals,
        origin=origin,
        shape=shape,
    )
    neurograph = init_immutables(
        neurograph,
        anisotropy=anisotropy,
        prune=prune,
        prune_depth=prune_depth,
        smooth=smooth
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
    smooth=True,
):
    """
    To do...
    """
    for swc_id in utils.listdir(neurograph.path, ext=".swc"):
        raw_swc = swc_utils.read_swc(os.path.join(neurograph.path, swc_id))
        swc_id = swc_id.replace(".0.swc", "")
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)
        if smooth:
            swc_dict = swc_utils.smooth(swc_dict)
        neurograph.generate_immutables(
            swc_id, swc_dict, prune=prune, prune_depth=prune_depth
        )
    return neurograph
