from torch import nn


from src.modules.training.models.res_bi_GRU import ResidualBiGRU


class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, out_size=2, n_layers=5, bidir=True, activation: str = None, flatten: bool = False, dropout: float = 0,
                 internal_layers: int = 1, model_name=''):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.flatten = flatten
        self.dropout = dropout

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [ResidualBiGRU(hidden_size, internal_layers=internal_layers, bidir=bidir, dropout=dropout)
             for _ in range(n_layers)
             ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

        if activation is None:
            self.activation = nn.Identity
        else:
            self.activation = nn.functional.__dict__[activation]

    def forward(self, x, h=None, use_activation=True):
        # if we are at the beginning of a sequence (no hidden state)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        # Flatten the (32,1440,3) to (32*1440, 3)

        if self.flatten:
            x = x.view(-1, x.shape[-1])

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i] if i == 0 else new_h[i - 1])
            new_h.append(new_hi)


        x = self.fc_out(x)
        if use_activation:
            x = self.activation(x)
        return x, new_h  
