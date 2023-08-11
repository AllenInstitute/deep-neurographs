"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Tests routines that build graph and generate features

"""

import os

import networkx as nx
import torch
import torch_geometric.transforms as T

from deep_neurographs import feature_extraction as extracter
from deep_neurographs import intake, net
from deep_neurographs import neurograph as ng
from deep_neurographs import train, utils


def train():
    # Data loader
    graph = build_neurographs()
    # node_features, edge_features = generate_features()

    # Cross validation


def build_neurographs():
    print("Building NeuroGraphs...")
    graphs = []
    for block_id in utils.listsubdirs(root_dir, keyword="block"):
        print("   " + block_id)
        swc_dir = os.path.join(root_dir, block_id)
        graphs.append(intake.build_neurograph(swc_dir))
    return graphs


def generate_features():
    pass


if __name__ == "__main__":
    # Paramaters
    anisotropy = [1.0, 1.0, 1.0]
    dataset = "653158"
    pred_id = "20230801_2steps_segmentation_filtered"
    root_dir = f"/home/jupyter/workspace/data/{dataset}/pred_swcs/{pred_id}"
    whole_brain = False

    # Main
    train()

    """
    # Feature extraction
    node_features = extracter.generate_node_features(
        supergraph, img=False, pointcloud=False
    )
    edge_features = extracter.generate_edge_features(supergraph)
    print("Generated node and edge features...")
    print("Number of node features:", node_features.shape[1])
    print("Number of edge features:", edge_features.shape[1])
    print("")


    # Initialize training data
    data, mistake_log = intake.init_data(
        supergraph, node_features, edge_features, bucket, mistake_log_path,
    )


    # Training parameters
    num_feats = node_features.shape[1]
    model = net.GCN(num_feats, num_feats // 2, num_feats // 4).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=False,
        ),
    ])


    # Train
    print("Training...")
    train_data, val_data, test_data = transform(data)
    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        loss = train.train(model, optimizer, criterion, train_data)
        val_auc = train.test(model, val_data)
        test_auc = train.test(model, test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    z = model.encode(test_data.x, test_data.edge_index)
    final_edge_index = model.decode_all(z)
    """
