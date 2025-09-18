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
    Convolutional neural network for 3D images.
    """

    def __init__(
        self,
        token_shape,
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
        token_shape : Tuple[int]
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
        flat_size = self._get_flattened_size(token_shape)
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

    def _get_flattened_size(self, token_shape):
        """
        Compute the flattened feature vector size after applying a sequence
        of convolutional and pooling layers on an input tensor with the given
        shape.

        Parameters
        ----------
        token_shape : Tuple[int]
            Shape of input image patch.

        Returns
        -------
        int
            Length of the flattened feature vector after the convolutions and
            pooling.
        """
        with torch.no_grad():
            x = torch.zeros(1, 2, *token_shape)
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
        in_channels=2,
        img_shape=(128, 128, 128),
        token_shape=(8, 8, 8),
        emb_dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
    ):
        # Call parent class
        super().__init__()

        # Class attributes
        self.token_shape = token_shape
        self.grid_size = [img_shape[i] // token_shape[i] for i in range(3)]

        # Transformer layers
        self.img_token_embed = ImageTokenEmbedding3D(
            in_channels, token_shape, emb_dim, img_shape
        )
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_dim, heads, mlp_dim)
                for _ in range(depth)
            ]
        )
        self.output_head = nn.Linear(
            emb_dim, np.prod(token_shape) * in_channels
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.img_token_embed(x)
        x = self.transformer(x)
        x = self.output_head(x)
        x = x.view(batch_size, -1, *self.token_shape)
        x = rearrange(
            x,
            "(b d h w) c pd ph pw -> b c (d pd) (h ph) (w pw)",
            b=batch_size,
            d=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2],
            pd=self.token_shape[0],
            ph=self.token_shape[1],
            pw=self.token_shape[2],
        )
        return x


class ImageTokenEmbedding3D(nn.Module):
    """
    A class for learning image token embeddings for transformer-based
    architectures.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer.
    emb_dim : int
        Dimension of the embedding space.
    pos_embedding : nn.Parameter
        Learnable position encoding.
    proj : nn.Conv3d
        Convolutional layer that generates a learnable projection of the
        tokens.
    token_shape : Tuple[int]
        Shape of each token (D, H, W).
    """

    def __init__(
        self, in_channels, token_shape, emb_dim, img_shape, dropout=0.1
    ):
        """
        Instantiates a ImageTokenEmbedding3D object.

        Parameters
        ----------
        in_channels : int
            Number of input channels in the image.
        token_shape : Tuple[int]
            Shape of each token (D, H, W).
        emb_dim : int
            Dimension of the embedding space.
        img_shape : Tuple[int]
            Shape of the input image (D, H, W).
        dropout : float, optional
            Dropout probability applied after adding positional embeddings.
            Default is 0.1.
        """
        # Call parent class
        super().__init__()

        # Class attributes
        self.emb_dim = emb_dim
        self.token_shape = token_shape

        # Embedding
        n_tokens = np.prod([s // ts for s, ts in zip(img_shape, token_shape)])
        self.pos_embedding = nn.Parameter(torch.randn(1, n_tokens, emb_dim))
        self.proj = nn.Conv3d(
            in_channels, emb_dim, kernel_size=token_shape, stride=token_shape
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass that converts an input image into a sequence of token
        embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Token embeddings of shape (B, N, E).
        """
        x = self.proj(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = x + self.pos_embedding
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block.

    Attributes
    ----------
    attn : nn.MultiheadAttention
        Multihead attention block.
    dropout : nn.Dropout
        Dropout layer.
    mlp : nn.Sequential
        Multi-layer perceptron.
    norm1 : nn.LayerNorm
        Applies layer normalization over a mini-batch of the inputs.
    norm2 : nn.LayerNorm
        Applied layer normalization over a mini-batch of the outputs of the
        multihead attention block.
    """

    def __init__(self, emb_dim, heads, mlp_dim, dropout=0.1):
        """
        Instantiates a TransformerEncoderBlock object.

        Parameters
        ----------
        emb_dim : int
            Dimension of the embedding space.
        heads : int
            Number of attention heads.
        mlp_dim : int
            Dimensionality of the hidden layer in the MLP.
        dropout : float, optional
            Dropout probability applied after attention and MLP layers.
            Default is 0.1.
        """
        # Call parent class
        super().__init__()

        # Attention head
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            emb_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = init_mlp(emb_dim, mlp_dim, emb_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass of the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, N, E).

        Returns
        -------
        x : torch.Tensor
            Output tensor of shape (B, N, E), same as input.
        """
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        x = self.dropout(x)
        return x


# --- Helpers ---
def init_mlp(input_dim, hidden_dim, output_dim, dropout=0.1):
    """
    Initializes a multi-layer perceptron (MLP).

    Parameters
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
