"""MultiResidualBiGRU top scoring solution from the parkinson challenge."""
from torch import Tensor, nn


class ResidualBiGRU(nn.Module):
    """ResidualBiGRU layer for a MultiResidualBiGRU model.

    :param hidden_size: Hidden size of the ResidualBiGRU layer
    :param n_layers: Number of layers of the ResidualBiGRU layer
    """

    def __init__(self, hidden_size: int, n_layers: int = 1) -> None:
        """Initialize a ResidualBiGRU layer.

        :param hidden_size: Hidden size of the ResidualBiGRU layer
        :param n_layers: Number of layers of the ResidualBiGRU layer
        """
        super(ResidualBiGRU, self).__init__()  # noqa: UP008

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.ln1 = nn.LayerNorm(hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of the ResidualBiGRU layer.

        :param x: Input Tensor
        :return: Output tensor
        """
        res, new_h = self.gru(x, None)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # Skip connection
        return res + x


class MultiResidualBiGRU(nn.Module):
    """MultiResidualBiGRU model.

    :param input_size: Input size of the model
    :param hidden_size: Hidden size of the model
    :param out_size: Out size of the model
    :param n_layers: Number of ResidualBiGRU layers
    """

    def __init__(self, input_size: int, hidden_size: int, out_size: int, n_layers: int) -> None:
        """Initialize a MultiResidualBiGRU model.

        :param input_size: Input size of model
        :param hidden_size: Hidden size of model
        :param out_size: Out size of the model
        :param n_layers: Number of ResidualBiGRU layers
        """
        super(MultiResidualBiGRU, self).__init__()  # noqa: UP008

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [ResidualBiGRU(hidden_size, n_layers=1) for _ in range(n_layers)],
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of the MultiResidualBiGRU.

        :param x: Input tensor
        :return: Tensor
        """
        x = self.fc_in(x.transpose(2, 1))
        x = self.ln(x)
        x = nn.functional.relu(x)

        for res_bigru in self.res_bigrus:
            x = res_bigru(x)

        x = x[:, -1, :]
        return self.fc_out(x)
