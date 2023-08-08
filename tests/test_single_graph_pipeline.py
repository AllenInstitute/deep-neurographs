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
from deep_neurographs import intake
from deep_neurographs import net
from deep_neurographs import train


if __name__ == "__main__":
    # Parmaters
    anisotropy = [1.0, 1.0, 1.0]
    bucket = "aind-msma-morphology-data"
    dataset = "651324"
    device = "cuda:0"
    block_id = "block_003"
    pred_id = "pred_3"


    # Initialize paths
    root_path = f"agrim-postprocessing-exps/data/{dataset}/{block_id}"
    swc_path = f"{root_path}/swcs/{pred_id}"
    mistake_log_path = f"{root_path}/mistake_logs/{pred_id}.txt"


    # Build graph
    supergraph = intake.build_graph(bucket, swc_path)
    print("Graph is built...")
    print("Number of nodes:", supergraph.num_nodes())
    print("Number of edges:", supergraph.num_edges())
    print("")

    
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