"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Tests routines that build graph and generate features

"""

import networkx as nx
import torch
import torch_geometric.transforms as T

from deep_neurographs import feature_extraction as extracter
from deep_neurographs import intake, net
from deep_neurographs import neurograph as ng
from deep_neurographs import train

if __name__ == "__main__":

    # Parameters
    max_mutable_degree=5
    max_mutable_dist=100.0
    prune=True
    prune_depth=16

    # Build graph from s3
    dataset = "651324"
    block_id = "block_003"
    pred_id = "pred_3"

    bucket = "aind-msma-morphology-data"
    s3_root = f"agrim-postprocessing-exps/data/{dataset}/{block_id}"
    swc_dir = f"{s3_root}/swcs/{pred_id}"

    neurograph_s3 = intake.build_neurograph(
        swc_dir,
        bucket=bucket,

        max_mutable_degree=max_mutable_degree,
        max_mutable_dist=max_mutable_dist,
        prune=prune,
        prune_depth=prune_depth,






    )
    print("Graph built from s3 data...")
    print("Number of nodes:", neurograph_s3.num_nodes())
    print("Number of immutable edges:", len(neurograph_s3.immutable_edges))
    print("Number of mutable edges:", len(neurograph_s3.mutable_edges))
    print("")

    # Build graph from local machine
    dataset = "653158"
    block_id = "block_003"
    pred_id = "20230801_2steps_segmentation_filtered"

    local_root = f"/home/jupyter/workspace/data/{dataset}/pred_swcs/{pred_id}"
    swc_dir = f"{local_root}/{block_id}"

    neurograph_local = intake.build_neurograph(
        swc_dir,

        max_mutable_degree=max_mutable_degree,
        max_mutable_dist=max_mutable_dist,
        prune=prune,
        prune_depth=prune_depth,






    )
    print("Graph built from local data...")
    print("Number of nodes:", neurograph_local.num_nodes())
    print("Number of immutable edges:", len(neurograph_local.immutable_edges))
    print("Number of mutable edges:", len(neurograph_local.mutable_edges))
    print("")
