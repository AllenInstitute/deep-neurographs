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
    accuracy = []
    accuracy_baseline = []
    data = dataset["dataset"]
    if "Net" in model_type:
        model.eval()
        hat_y = []
        for batch in DataLoader(data, batch_size=32, shuffle=False):
            # Run model
            with torch.no_grad():
                x_i = batch["inputs"]
                hat_y_i = sigmoid(model(x_i))

            # Postprocess
            hat_y_i = np.array(hat_y_i)
            y_i = np.array(batch["targets"])
            hat_y.extend(hat_y_i.tolist())
            accuracy_baseline.extend((y_i > 0).tolist())
            accuracy.extend(((hat_y_i > 0.5) == (y_i > 0)).tolist())
        accuracy = np.mean(accuracy)
        accuracy_baseline = np.sum(accuracy_baseline) / len(accuracy_baseline)
        print("Accuracy +/-:", accuracy - accuracy_baseline)
    else:
        hat_y = model.predict_proba(data["inputs"])[:, 1]
    return np.array(hat_y)
