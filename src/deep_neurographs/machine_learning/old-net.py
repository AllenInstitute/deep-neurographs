"""
Created on Mon July 27 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Graph neural network architectures

"""

import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# --- Generate training data ---
def init_data(
    supergraph,
    node_features,
    edge_features,
    bucket,
    file_key,
    access_key_id=None,
    secret_access_key=None,
):
    """
    To do...
    """
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(list(supergraph.edges()), dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float)
    edge_label_index = None  # target labels
    data = Data(
        x=x,
        edge_index=edge_index.t().contiguous(),
        edge_label_index=edge_label_index,
        edge_attr=edge_features,
    )
    return data


"""
def train(model, optimizer, criterion, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),
        method="sparse",
    )

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index], dim=-1
    )
    edge_label = torch.cat(
        [
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())



from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False),
    ])
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'Planetoid'
    )
    dataset = Planetoid(path, name='Cora', transform=transform)

    # After applying the `RandomLinkSplit` transform, the data is transformed
    # from a data object to list of tuples (train_data, val_data, test_data),
    # with each element representing the corresponding split.
    train_data, val_data, test_data = dataset[0]
    print(train_data)

    model = Net(dataset.num_features, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = final_test_auc = 0
    for epoch in range(1, 101):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    z = model.encode(test_data.x, test_data.edge_index)
    final_edge_index = model.decode_all(z)
"""
