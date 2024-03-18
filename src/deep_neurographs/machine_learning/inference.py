"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for running inference on models that classify edge proposals.

"""

import numpy as np
import torch
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
                y_pred_i = sigmoid(model(x_i))
            y_pred.extend(np.array(y_pred_i).tolist())
    else:
        y_pred = model.predict_proba(dataset["inputs"])[:, 1]
    return np.array(y_pred)
