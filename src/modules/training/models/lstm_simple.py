from torch import nn


class LSTMSimple(nn.Module):
    ## Pass
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMSimple, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim = output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out