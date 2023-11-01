from random import sample
from deep_neurographs import utils
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import roc_auc_score
import torchio as tio

import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import negative_sampling
from torcheval.metrics.functional import (
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
)

BATCH_SIZE = 32
NUM_WORKERS = 0
SHUFFLE = True


# Training
def get_kfolds(train_data, k):
    folds = []
    samples = set(train_data)
    num_samples = int(np.floor(len(train_data) / k))
    assert num_samples > 0, "Sample size is too small for {}-folds".format(k)
    for i in range(k):
        samples_i = sample(samples, num_samples)
        samples = samples.difference(samples_i)
        folds.append(samples_i)
        if num_samples > len(samples):
            break
    return folds


def train_network(dataset, net, max_epochs=100):
    # Load data
    train_set, valid_set = random_split(dataset)
    train_loader = DataLoader(
        train_set,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
    )
    valid_loader = DataLoader(
        valid_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE
    )

    # Fit model
    model = LitNeuralNet(net)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val_f1", mode="max"
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, valid_loader)
    return model


def random_split(train_set, train_ratio=0.8):
    train_set_size = int(len(train_set) * train_ratio)
    valid_set_size = len(train_set) - train_set_size
    return torch_data.random_split(train_set, [train_set_size, valid_set_size])


def eval_network(X, model, threshold=0.5):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    y_pred = model.net(X)
    return np.array(y_pred > threshold, dtype=int)


# Custom Datasets
class ProposalDataset(Dataset):
    def __init__(self, inputs, labels, transform=None, target_transform=None):
        self.inputs = inputs.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"inputs": self.inputs[idx], "labels": self.labels[idx]}


class ImgProposalDataset(Dataset):
    def __init__(self, inputs, labels, transform=True):
        self.inputs = self.reformat(inputs)
        self.labels = self.reformat(labels)
        if transform:
            self.transform = Augmentator()
        self.transform_bool = True if transform else False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.transform_bool:
            inputs = utils.normalize(self.inputs[idx])
            inputs = self.transform.run(inputs)
        else:
            inputs = self.inputs[idx]
        return {"inputs": inputs, "labels": self.labels[idx]}

    def reformat(self, x):
        return np.expand_dims(x, axis=1).astype(np.float32)


class Augmentator:
    def __init__(self):
        self.blur = tio.RandomBlur(std=(0, 0.5)) # 1
        self.noise = tio.RandomNoise(std=(0, 0.03))
        self.elastic = tio.RandomElasticDeformation(max_displacement=10)
        self.apply_geometric = tio.Compose({
            #tio.RandomFlip(axes=(0, 1, 2)),
            tio.RandomAffine(degrees=30, scales=(0.8, 1)),
        })

    def run(self, arr):
        arr = self.blur(arr)
        arr = self.noise(arr)
        #arr = self.elastic(arr)
        arr = self.apply_geometric(arr)
        return arr


# Neural Network Training
class LitNeuralNet(pl.LightningModule):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, batch):
        x = self.get_example(batch, "inputs")
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        X = self.get_example(batch, "inputs")
        y = self.get_example(batch, "labels")
        y_hat = self.net(X)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss)
        self.compute_stats(y_hat, y, prefix="train_")
        return loss

    def validation_step(self, batch, batch_idx):
        X = self.get_example(batch, "inputs")
        y = self.get_example(batch, "labels")
        y_hat = self.net(X)
        self.compute_stats(y_hat, y, prefix="val_")

    def compute_stats(self, y_hat, y, prefix=""):
        y_hat = torch.flatten(y_hat)
        y = torch.flatten(y).to(torch.int)
        self.log(prefix + "accuracy", binary_accuracy(y_hat, y))
        self.log(prefix + "precision", binary_precision(y_hat, y))
        self.log(prefix + "recall", binary_recall(y_hat, y))
        self.log(prefix + "f1", binary_f1_score(y_hat, y))

    def get_example(self, batch, key):
        return batch[key]


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
