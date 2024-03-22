"""Module containing simple gru model."""
from torch import Tensor, nn


class GRUTimeSeriesClassifier(nn.Module):
    """Classifier using GRU architecture.

    :param num_classes: Number of input classes
    :param input_dim: Input dimension (length of sequence)
    :param hidden_dim: Hidden dimension
    :param gru_layers: Number of gru_layers
    :param bidirectional: Whether gru should be bidirectional
    :param dropout: Dropout of GRU
    """

    def __init__(self, num_classes: int, input_dim: int, hidden_dim: int = 128, gru_layers: int = 2, *, bidirectional: bool = False, dropout: float = 0.1) -> None:
        """Initialize class.

        :param num_classes: Number of input classes
        :param input_dim: Input dimension (length of sequence)
        :param hidden_dim: Hidden dimension
        :param gru_layers: Number of gru_layers
        :param bidirectional: Whether gru should be bidirectional
        :param dropout: Dropout of GRU
        """
        super(GRUTimeSeriesClassifier, self).__init__()  # noqa: UP008
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=gru_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward input through model.

        :param x: Input tensor
        :return: Output tensor
        """
        # x: (batch_size, seq_length, input_dim)
        x = x.permute(0,2,1)
        gru_out, _ = self.gru(x)
        if self.bidirectional:
            # Use the concatenated last hidden states of both directions
            gru_out = gru_out[:, -1, :]
        else:
            # Use the last hidden state
            gru_out = gru_out[:, -1, :]
        out = self.dropout(gru_out)
        out = self.fc(out)
        return out
