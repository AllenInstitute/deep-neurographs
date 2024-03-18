"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference on models that classify edge proposals.

"""

import torch
import numpy as np
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader


def predict(dataset, model, model_type):
    dataset = dataset["dataset"]
    if "Net" in model_type:
        model.eval()
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        y_pred = []
        for batch in dataloader:
            with torch.no_grad():
                x_i = batch["inputs"]
                y_i = batch["targets"]
                y_pred_i = sigmoid(model(x_i))
                y_pred.extend(np.array(y_pred_i).tolist())
                print(((np.array(y_pred_i) > 0.5) == y_i) / len(y_i))
    else:
        model.predict_proba(dataset["inputs"])[:, 1]
    return np.array(y_pred)
