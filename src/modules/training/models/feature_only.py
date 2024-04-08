from torch import nn


class FeatureOnly(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super(FeatureOnly, self).__init__()
        self.activation = nn.SiLU()
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x_deep, x_manual):
        x = x_manual
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)