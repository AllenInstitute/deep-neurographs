"""
Created on Sat July 15 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for vision models that perform image classification tasks within
GraphTrace pipelines.

"""

from einops import rearrange

import numpy as np
import torch
import torch.nn as nn


# --- CNNs ---
class CNN3D(nn.Module):
    """
    Convolutional neural network that classifies edge proposals given an image
    patch.
    """

    def __init__(
        self,
        patch_shape,
        output_dim=1,
        dropout=0.1,
        n_conv_layers=5,
        n_feat_channels=16,
        use_double_conv=True
    ):
        """
        Constructs a ConvNet object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of input image patch.
        output_dim : int, optional
            Dimension of output. Default is 1.
        dropout : float, optional
            Fraction of values to randomly drop during training. Default is
            0.1.
        n_conv_layers : int, optional
            Number of convolutional layers. Default is 5.
        use_double_conv : bool, optional
            Indication of whether to use double convolution. Default is True.
        """
        # Call parent class
        nn.Module.__init__(self)

        # Class attributes
        self.dropout = dropout
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.use_double_conv = use_double_conv

        # Dynamically build convolutional layers
        layers = []
        in_channels = 2
        out_channels = n_feat_channels
        for i in range(n_conv_layers):
            layers.append(self._init_conv_layer(in_channels, out_channels, 3))
            in_channels = out_channels
            out_channels *= 2
        self.conv_layers = nn.ModuleList(layers)

        # Output layer
        flat_size = self._get_flattened_size(patch_shape)
        self.output = nn.Sequential(
            init_mlp(flat_size, flat_size * 2, flat_size // 2),
            init_mlp(flat_size // 2, flat_size, flat_size // 4),
            init_mlp(flat_size // 4, flat_size // 2, output_dim),
        )

        # Initialize weights
        self.apply(self.init_weights)

    def _init_conv_layer(
        self, in_channels, out_channels, kernel_size
    ):
        """
        Initializes a convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of channels that are input to this convolutional layer.
        out_channels : int
            Number of channels that are output from this convolutional layer.
        kernel_size : int
            Size of kernel used on convolutional layers.

        Returns
        -------
        torch.nn.Sequential
            Sequence of operations that define this layer.
        """

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding="same"
                ),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
            )

        if self.use_double_conv:
            return nn.Sequential(
                conv_block(in_channels, out_channels),
                conv_block(out_channels, out_channels),
            )
        else:
            return conv_block(in_channels, out_channels)

    def _get_flattened_size(self, patch_shape):
        """
        Compute the flattened feature vector size after applying a sequence
        of convolutional and pooling layers on an input tensor with the given
        shape.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of input image patch.

        Returns
        -------
        int
            Length of the flattened feature vector after the convolutions and
            pooling.
        """
        with torch.no_grad():
            x = torch.zeros(1, 2, *patch_shape)
            for conv in self.conv_layers:
                x = conv(x)
                x = self.pool(x)
            return x.view(1, -1).size(1)

    @staticmethod
    def init_weights(m):
        """
        Initializes the weights and biases of a given PyTorch layer.

        Parameters
        ----------
        m : nn.Module
            PyTorch layer or module.
        """
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Passes an input vector "x" through this neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input vector of features.

        Returns
        -------
        x : torch.Tensor
            Output of the neural network.
        """
        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
            x = self.pool(x)

        # Output layer
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


# --- Transformers ---
class ViT3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        img_shape=(64, 64, 64),
        patch_shape=(8, 8, 8),
        emb_size=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
    ):
        # Call parent class
        super().__init__()

        # Class attributes
        self.patch_shape = patch_shape
        self.grid_size = [img_shape[i] // patch_shape[i] for i in range(3)]

        # Transformer Layers
        self.patch_embed = PatchEmbedding3D(
            in_channels, patch_shape, emb_size, img_shape
        )
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_size, heads, mlp_dim)
                for _ in range(depth)
            ]
        )
        self.output_head = nn.Linear(
            emb_size, np.prod(patch_shape) * in_channels
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.output_head(x)
        x = x.view(batch_size, -1, *self.patch_shape)
        x = rearrange(
            x,
            "(b d h w) c pd ph pw -> b c (d pd) (h ph) (w pw)",
            b=batch_size,
            d=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2],
            pd=self.patch_shape[0],
            ph=self.patch_shape[1],
            pw=self.patch_shape[2],
        )
        return x


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, patch_shape, emb_size, img_shape, dropout=0.1):
        # Call parent class
        super().__init__()

        # Class attributes
        self.patch_shape = patch_shape
        self.emb_size = emb_size

        # Embedding
        n_patches = np.prod([img_shape[i] // patch_shape[i] for i in range(3)])
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, emb_size))
        self.proj = nn.Conv3d(
            in_channels, emb_size, kernel_size=patch_shape, stride=patch_shape
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = x + self.pos_embedding
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, heads, mlp_dim, dropout=0.1):
        # Call parent class
        super().__init__()

        # Attention head
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# --- Helpers ---
def init_mlp(input_dim, hidden_dim, output_dim, dropout=0.1):
    """
    Initializes a multi-layer perceptron (MLP).

    Parametersƒ
    ----------
    input_dim : int
        Dimension of input feature vector.
    hidden_dim : int
        Dimension of embedded feature vector.
    output_dim : int
        Dimension of output feature vector.
    dropout : float, optional
        Fraction of values to randomly drop during training. Default is 0.1.

    Returns
    -------
    mlp : nn.Sequential
        Multi-layer perception network.
    """
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, output_dim)
    )
    return mlp
