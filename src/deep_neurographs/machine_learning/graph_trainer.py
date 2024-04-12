"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training graph neural networks that classify edge proposals.

"""

import torch
from random import sample, shuffle


LR = 1e-3
N_EPOCHS = 100
TEST_PERCENT = 0.15
WEIGHT_DECAY = 5e-4


def run_on_graph(model, graph_data):
      pass


def run_on_graphs(
    model,
    criterion,
    graph_datasets,
    lr=LR,
    n_epochs=N_EPOCHS,
    weight_decay=WEIGHT_DECAY,
):
    # Initializations
    graph_ids = list(graph_datasets.keys())
    model.train()
    model.to("cuda:0")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Train
    train_ids, test_ids = train_test_split(list(graph_datasets.keys()))
    for epoch in range(n_epochs):
         for graph_id in train_ids:
            loss, optimizer = train(
                model,
                criterion,
                optimizer,
                graph_datasets[graph_id].data,
            )
    return model


def train(model, criterion, optimizer, graph_data):
    # Move data to gpu
    x = graph_data.x.to("cuda:0", dtype=torch.float32)
    y = graph_data.y.to("cuda:0", dtype=torch.float32)
    edge_index = graph_data.edge_index.to("cuda:0")

    # Forward pass
    n = y.size(0)
    optimizer.zero_grad()
    preds = model(x, edge_index)
    loss = criterion(preds[0:n, 0], y)

    # Backward pass
    loss.backward()
    optimizer.step()
    return loss, optimizer


def validate(model, graph_data):
    model.eval()
            x = graph_data.x.to("cuda:0", dtype=torch.float32)
    y = graph_data.y.to("cuda:0", dtype=torch.float32)
    edge_index = graph_data.edge_index.to("cuda:0")
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc


# -- utils --
def shuffler(my_list):
    """
    Shuffles a list of items.

    Parameters
    ----------
    my_list : list
        List to be shuffled.

    Returns
    -------
    my_list : list
        Shuffled list.

    """
    shuffle(my_list)
    return my_list


def train_test_split(graph_ids):
    n_test_examples = int(len(graph_ids) * TEST_PERCENT)
    test_ids = sample(graph_ids, n_test_examples)
    train_ids = list(set(graph_ids) - set(test_ids))
    return train_ids, test_ids
