"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Builds neurograph for neuron reconstruction.

"""

import os

import torch
from torch_geometric.data import Data

from deep_neurographs.neurograph import NeuroGraph
from deep_neurographs import s3_utils, swc_utils, utils


# --- Build graph ---
def build_neurograph(
    swc_dir,
    anisotropy=[1.0, 1.0, 1.0],
    bucket=None,
    access_key_id=None,
    secret_access_key=None,
    max_mutable_degree=5,
    max_mutable_dist=50.0,
    prune=True,
    prune_depth=16,
    origin=None,
    shape=None,
):
    """
    Builds a neurograph from a directory of swc files, where each swc
    represents a neuron and these neurons are assumed to be near each
    other.

    """
    neurograph = NeuroGraph(swc_dir, origin=origin, shape=shape)
    if bucket is not None:
        neurograph = init_immutables_from_s3(
            neurograph,
            bucket,
            anisotropy=anisotropy,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            prune=prune,
            prune_depth=prune_depth,
        )
    else:
        neurograph = init_immutables_from_local(
            neurograph,
            anisotropy=anisotropy,
            prune=prune,
            prune_depth=prune_depth,
        )
    neurograph.generate_mutables(
        max_degree=max_mutable_degree, max_dist=max_mutable_dist
    )
    return neurograph


def init_immutables_from_s3(
    neurograph,
    bucket,
    anisotropy=[1.0, 1.0, 1.0],
    access_key_id=None,
    secret_access_key=None,
    prune=True,
    prune_depth=16,
    smooth=False,
):
    """
    To do...
    """
    s3_client = s3_utils.init_session(
        access_key_id=access_key_id, secret_access_key=secret_access_key
    )
    swc_files = s3_utils.listdir(bucket, neurograph.path, s3_client, ext=".swc")
    for file_key in swc_files:
        swc_id = file_key.split("/")[-1].replace(".swc", "")
        raw_swc = s3_utils.read_from_s3(bucket, file_key, s3_client)
        swc_dict = swc_utils.parse(raw_swc, anisotropy=anisotropy)
        if smooth:
            swc_dict = swc_utils.smooth(swc_dict)
        neurograph.generate_immutables(
            swc_id, swc_dict, prune=prune, prune_depth=prune_depth
        )
    return neurograph


def init_immutables_from_local(
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
