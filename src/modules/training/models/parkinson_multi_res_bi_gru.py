from torch import nn


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(ResidualBiGRU, self).__init__()

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

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h  # log probabilities + hidden state


class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [ResidualBiGRU(hidden_size, n_layers=1) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        # if we are at the beginning of a sequence (no hidden state)
        # if h is None:
        #     # (re)initialize the hidden state
        #     h = [None for _ in range(self.n_layers)]
        x = self.fc_in(x.transpose(2, 1))
        x = self.ln(x)
        x = nn.functional.relu(x)

        for i, res_bigru in enumerate(self.res_bigrus):
            x, _ = res_bigru(x, None)

        x = x[:, -1, :]
        x = self.fc_out(x)

        return x  # , new_h  # log probabilities + hidden states
