"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Tests routines that build graph and generate features

"""

from deep_neurographs import feature_extraction as extracter
from deep_neurographs import gcn, intake

"""
To do...
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
    mistake_log_path = f"{root_path}/mistake_logs/{pred_id}.txt"

    # Main
    supergraph = intake.build_graph(bucket, swc_path)
    print("Graph is built")

    node_features = extracter.generate_node_features(
        supergraph, img=False, pointcloud=False
    )
    edge_features = extracter.generate_edge_features(supergraph)
    print("Generated node and edge features")

    training_data = gcn.init_training_data(
        supergraph, node_features, edge_features
    )
    target_edges = intake.get_target_edges(supergraph, bucket, mistake_log_path)
