"""
Created on Sat April 12 11:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for training graph neural networks that classify edge proposals.

"""

from random import sample, shuffle

import torch
from torch.nn.functional import sigmoid

LR = 1e-3
N_EPOCHS = 300
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
    model.to("cuda:0")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Main
    accuracy = []
    train_ids, test_ids = train_test_split(list(graph_datasets.keys()))
    for epoch in range(n_epochs):
        # Train
        model.train()
        for graph_id in train_ids:
            loss, optimizer = train(
                model, criterion, optimizer, graph_datasets[graph_id].data
            )

        # Test
        model.eval()
        accuracy_i = 0
        for graph_id in test_ids:
            accuracy_i += validate(model, graph_datasets[graph_id].data)
        accuracy.append(accuracy_i / len(test_ids))
        if epoch % 10 == 0:
            print("Accuracy +/-:", accuracy[-1])
    return model


def train(model, criterion, optimizer, graph_data):
    # Forward pass
    x, y, edge_index = toGPU(graph_data)
    optimizer.zero_grad()
    hat_y = model(x, edge_index)
    hat_y = truncate(hat_y, y)

    # Backward pass
    loss = criterion(hat_y, y)
    loss.backward()
    optimizer.step()
    return loss, optimizer


def validate(model, graph_data):
    # Initializations
    x, y, edge_index = toGPU(graph_data)
    hat_y = model(x, edge_index)
    hat_y = truncate(hat_y, y)

    # Compute accuracy
    preds = get_predictions(hat_y)
    correct = preds == y
    acc = float(correct.sum()) / y.size(0)
    return acc - y.sum() / y.size(0)


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
    n_test_examples = 1  # int(len(graph_ids) * TEST_PERCENT)
    test_ids = sample(graph_ids, n_test_examples)
    train_ids = list(set(graph_ids) - set(test_ids))
    return train_ids, test_ids


def toGPU(graph_data):
    x = graph_data.x.to("cuda:0", dtype=torch.float32)
    y = graph_data.y.to("cuda:0", dtype=torch.float32)
    edge_index = graph_data.edge_index.to("cuda:0")
    return x, y, edge_index


def truncate(hat_y, y):
    """
    Truncates "hat_y" so that this tensor has the same shape as "y". Note this
    operation removes the predictions corresponding to branches so that loss
    is computed over proposals.

    Parameters
    ----------
    hat_y : torch.Tensor
        Tensor to be truncated.
    y : torch.Tensor
        Tensor used as a reference.

    Returns
    -------
    torch.Tensor
        Truncated "hat_y".

    """
    return hat_y[0: y.size(0), 0]


def get_predictions(hat_y, threshold=0.5):
    return sigmoid(hat_y) > threshold
