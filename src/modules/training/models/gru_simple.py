from torch import nn
import torch.nn.functional as F


class GRUTimeSeriesClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim=128, gru_layers=2, bidirectional=False, dropout=0.1):
        super(GRUTimeSeriesClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=gru_layers, 
                          batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        gru_out, _ = self.gru(x)
        if self.bidirectional:
            # Use the concatenated last hidden states of both directions
            gru_out = gru_out[:, -1, :]
        else:
            # Use the last hidden state
            gru_out = gru_out[:, -1, :]
        out = self.dropout(gru_out)
        out = self.fc(out)
        return self.softmax(out)