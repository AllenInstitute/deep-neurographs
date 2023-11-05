"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Neural network architectures.

"""

import torch
from torch import nn


class FeedForwardNet(nn.Module):
    def __init__(self, num_features):
        nn.Module.__init__(self)
        self.fc1 = self._init_fc_layer(num_features, num_features)
        self.fc2 = self._init_fc_layer(num_features, num_features)
        self.output = nn.Sequential(nn.Linear(num_features, 1))

    def _init_fc_layer(self, D_in, D_out):
        fc_layer = nn.Sequential(
            nn.Linear(D_in, D_out), nn.LeakyReLU(), nn.Dropout(p=0.25)
        )
        return fc_layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = self._init_conv_layer(2, 4)
        self.conv2 = self._init_conv_layer(4, 4)
        self.output = nn.Sequential(
            nn.Linear(10976, 64), nn.LeakyReLU(), nn.Linear(64, 1)
        )

    def _init_conv_layer(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )
        return conv_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(vectorize(x))
        return x


class MultiModalNet(nn.Module):
    def __init__(self, num_features):
        nn.Module.__init__(self)
        self.fc1 = self._init_fc_layer(num_features, num_features)
        self.fc2 = self._init_fc_layer(num_features, num_features)
        self.conv1 = self._init_conv_layer(2, 4)
        self.conv2 = self._init_conv_layer(4, 4)
        self.linear =  nn.Sequential(nn.Linear(10976, 64), nn.LeakyReLU())
        self.output = FeedForwardNet(64 + num_features)

    def _init_fc_layer(self, D_in, D_out):
        fc_layer = nn.Sequential(
            nn.Linear(D_in, D_out), nn.LeakyReLU(), nn.Dropout(p=0.25)
        )
        return fc_layer

    def _init_conv_layer(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )
        return conv_layer

    def forward(self, x):
        # Feedforward arm
        x0 = self.fc1(x[0])
        x0 = self.fc2(x0)

        # CNN arm
        x1 = self.conv1(x[1])
        x1 = self.conv2(x1)
        x1 = self.linear(vectorize(x1))

        # Fusion
        x = torch.cat((x0, x1), axis=1)
        x = self.output(x)
        return x


def init_weights(net):
    for module in net.modules():
        if isinstance(module, nn.Conv3d):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)


def vectorize(x):
    return x.view(x.size(0), -1)