"""Module containing simple implementation of LSTM."""
from torch import Tensor, nn


class LSTMSimple(nn.Module):
    """Simple LSTM classifier.

    :param input_dim: Input dimension
    :param hidden_dim: Hidden dimension
    :param output_dim: Output dimension/class
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        """Initialize lstmsimple model.

        :param input_dim: Input dimension
        :param hidden_dim: Hidden dimension
        :param output_dim: Output dimension/class
        """
        super(LSTMSimple, self).__init__()  # noqa: UP008
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x: Tensor) -> Tensor:
        """Pass the input through the model.

        :param x: Input tensor
        :return: output tensor
        """
        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.softmax(out)
