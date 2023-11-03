import torch
from torch import nn


class FeedForwardNet(nn.Module):
    def __init__(self, num_features):
        nn.Module.__init__(self)
        self.fc1 = self._init_fc_layer(num_features, num_features)
        self.fc2 = self._init_fc_layer(num_features, num_features // 2)
        self.output = nn.Sequential(nn.Linear(num_features // 2, 1))

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
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


class MultiModalNet(nn.Module):
    def __init__(self, feature_vec_shape, img_patch_shape):
        self.fnn_arm = FeedForwardNet()
        self.cnn_arm = ConvNet()
        self.output = FeedForwardNet()

    def forward(self, feature_vec, img_chunk):
        x = torch.cat(
            (self.fnn_arm(feature_vec), self.cnn_arm(img_chunk)), axis=0
        )
        x = self.output(x)
        return x


def init_weights(net):
    for module in net.modules():
        if isinstance(module, nn.Conv3d):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
