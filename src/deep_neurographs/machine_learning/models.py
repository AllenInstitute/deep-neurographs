"""
Created on Sat November 04 15:30:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Neural network architectures that learn to classify edge proposals.

"""

from torch import nn

import torch


class FeedForwardNet(nn.Module):
    """
    Feedforward neural network that classifies edge proposals given a feature
    vector.

    """

    def __init__(self, num_features):
        """
        Constructs a FeedForwardNet object.

        Parameters
        ----------
        num_features : torch.array
            Number of features in the input feature vector.

        Returns
        -------
        None

        """
        nn.Module.__init__(self)
        self.fc1 = self._init_fc_layer(num_features, 2 * num_features)
        self.fc2 = self._init_fc_layer(2 * num_features, num_features // 4)
        self.output = nn.Linear(num_features // 4, 1)

    def _init_fc_layer(self, D_in, D_out):
        """
        Initializes a fully connected linear layer.

        Parameters
        ----------
        D_in : int
            Length of vector that is input to this layer.
        D_out : int
            Length of vector that is output from this layer.

        Returns
        -------
        torch.nn.Sequential
            Sequence of operations that define this layer.

        """
        fc_layer = nn.Sequential(
            nn.Linear(D_in, D_out), nn.LeakyReLU(), nn.Dropout(p=0.25)
        )
        return fc_layer

    def forward(self, x):
        """
        Passes an input vector "x" through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        torch.Tensor
            Output of neural network.

        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

    def model_type(self):
        return "FeedForwardNet"


class ConvNet(nn.Module):
    """
    Convolutional neural network that classifies edge proposals given an image
    patch.

    """

    def __init__(self, patch_shape, output_dim):
        """
        Constructs a ConvNet object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        nn.Module.__init__(self)
        self.conv1 = self._init_conv_layer(2, 32)
        self.conv2 = self._init_conv_layer(32, 64)
        self.output = nn.Sequential(
            nn.Linear(85184, 64),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _init_conv_layer(self, in_channels, out_channels):
        """
        Initializes a convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of channels that are input to this convolutional layer.
        out_channels : int
            Number of channels that are output from this convolutional layer.

        Returns
        -------
        torch.nn.Sequential
            Sequence of operations that define this layer.

        """
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        return conv_layer

    def forward(self, x):
        """
        Passes an input vector "x" through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        torch.Tensor
            Output of neural network.

        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(vectorize(x))
        return x


class MultiModalNet(nn.Module):
    """
    Multimodal neural network that classifies edge proposals given an image
    chunk and feature vector.

    """

    def __init__(self, num_features):
        """
        Constructs a MultiModalNet object.

        Parameters
        ----------
        num_features : int
            Number of features in the input feature vector.

        Returns
        -------
        None

        """
        nn.Module.__init__(self)
        self.fc1 = self._init_fc_layer(num_features, num_features)
        self.fc2 = self._init_fc_layer(num_features, num_features)
        self.conv1 = self._init_conv_layer(2, 4)
        self.conv2 = self._init_conv_layer(4, 4)
        self.linear = nn.Sequential(nn.Linear(10976, 64), nn.LeakyReLU())
        self.output = FeedForwardNet(64 + num_features)

    def _init_fc_layer(self, D_in, D_out):
        """
        Initializes a fully connected linear layer.

        Parameters
        ----------
        D_in : int
            Length of vector that is input to this layer.
        D_out : int
            Length of vector that is output from this layer.

        Returns
        -------
        torch.nn.Sequential
            Sequence of operations that define this layer.

        """
        fc_layer = nn.Sequential(
            nn.Linear(D_in, D_out), nn.LeakyReLU(), nn.Dropout(p=0.25)
        )
        return fc_layer

    def _init_conv_layer(self, in_channels, out_channels):
        """
        Initializes a convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of channels that are input to this convolutional layer.
        out_channels : int
            Number of channels that are output from this convolutional layer.

        Returns
        -------
        torch.nn.Sequential
            Sequence of operations that define this layer.

        """
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
        """
        Passes an input vector "x" through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        torch.Tensor
            Output of neural network.

        """
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

    def model_type(self):
        return "MultiModalNet"


def init_weights(net):
    """
    Initializes the weights of a neural network "net" by using Xavier
    normal initialization.

    Parameters
    ----------
    net : nn.Module
        Neural network.

    Returns
    -------
    None

    """
    for module in net.modules():
        if isinstance(module, nn.Conv3d):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)


def vectorize(tensor):
    """
    Transforms a tensor into a vector.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be transformed.

    Returns
    -------
    torch.Tensor
        Vectorized input tensor.

    """
    return tensor.view(tensor.size(0), -1)
