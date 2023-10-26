from random import sample

import lightning.pytorch as pl
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch.utils.data import Dataset
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)


# Cross Validation
def get_kfolds(train_data, k):
    folds = []
    samples = set(train_data)
    num_samples = int(np.floor(len(train_data) / k))
    assert num_samples > 0, "Sample size is too small for {}-folds".format(k)
    for i in range(k):
        if i < k - 1:
            samples_i = sample(samples, num_samples)
            samples = samples.difference(samples_i)
            folds.append(set(samples_i))
        else:
            folds.append(samples)
    return folds


# Neural Network Training
class EdgeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"data": self.data[idx], "label": self.labels[idx]}


class LitNeuralNet(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = self.get_example(batch, "data")
        y = self.get_example(batch, "label")
        y_hat = self.net(x)
        return F.mse_loss(y_hat, y)

    def test_step(self, batch, batch_idx):
        x = self.get_example(batch, "data")
        y = self.get_example(batch, "label")
        y_hat = self.net(x)
        self.compute_stats(y_hat, y)

    def compute_stats(self, y_hat, y):
        y_hat = torch.flatten(y_hat)
        y = torch.flatten(y).to(torch.int)
        self.log("accuracy", binary_accuracy(y_hat, y))
        self.log("precision", binary_precision(y_hat, y))
        self.log("recall", binary_recall(y_hat, y))
        self.log("f1", binary_f1_score(y_hat, y))

    def get_example(self, batch, key):
        return batch[key].view(batch[key].size(0), -1)


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


"""
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
    
    # After applying the `RandomLinkSplit` transform, the data is transformed from
    # a data object to a list of tuples (train_data, val_data, test_data), with
    # each element representing the corresponding split.
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
