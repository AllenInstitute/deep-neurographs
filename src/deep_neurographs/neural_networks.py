from torch import nn


class FeedFowardNet(nn.Module):
    def __init__(self, num_features, depth=3):
        nn.Module.__init__(self)

        # Parameters
        assert depth < num_features
        self.depth = depth
        self.num_features = num_features

        # Layers
        print("Network Architecture...")
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=0.2)
        for d in range(self.depth):
            D_in = num_features // max(d, 1)
            D_out = num_features // (d + 1)
            self.add_fc_layer(d, D_in, D_out)
        self.last_fc = nn.Linear(D_out, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for d in range(self.depth):
            fc_d = getattr(self, "fc{}".format(d))
            x = self.activation(self.dropout(fc_d(x)))
        x = self.last_fc(x)
        return self.sigmoid(x)

    def add_fc_layer(self, d, D_in, D_out):
        setattr(self, "fc{}".format(d), nn.Linear(D_in, D_out))
        print("   {} --> {}".format(D_in, D_out))


class ConvNet(nn.Module):
    def __init__(self, input_dims, depth=3):
        pass


class MultiModalNet(nn.Module):
    def __init__(self, feature_vec_shape, img_patch_shape):
        pass
