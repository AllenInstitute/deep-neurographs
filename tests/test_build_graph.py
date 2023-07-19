"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Tests routines that build graph and generate features

"""

import networkx as nx
from deep_neurographs import feature_extraction as extracter, gcn, intake



"""
Sketch for build_graph

    # X - read files

    # -- Node creation --
    # X - init graph
    # X - init dict for (x, y, z) --> new id
    # X - init upd_obj_id dict
    # X - loop through swc files
    # X - create node <-- function call
    # X - assign new id
    # X - generate features <-- function call
    # X - store skeleton in sparse dict (for edge creation)
    
    # -- Edge creation --
    # X - KDTree <-- function call
    # generate edge features
    
    # -- Generate ground truth connectivity --
    # read mistake log
    # generate target adjacency matrix
"""

if __name__ == "__main__":
    # Parmaters
    anisotropy = [1.0, 1.0, 1.0]
    bucket = "aind-msma-morphology-data"
    dataset = "651324"
    block_id = "block_003"
    pred_id = "pred_3"

    # Initializations
    root_path = f"agrim-postprocessing-exps/data/{dataset}/{block_id}"
    swc_path = f"{root_path}/swcs/{pred_id}"
    mistake_log_path = f"{root_path}/mistake_logs/{pred_id}.json"

    # Main
    supergraph = intake.build_graph(bucket, swc_path)
    print("Graph is built")

    node_features = extracter.generate_node_features(supergraph, img=False, pointcloud=False)
    edge_features = extracter.generate_edge_features(supergraph)
    print("Generated node and edge features")

    training_data = gcn.init_training_data(supergraph, node_features, edge_features)
    target_edges = intake.get_target_edges(mistake_log_path)
    print(training_data)
