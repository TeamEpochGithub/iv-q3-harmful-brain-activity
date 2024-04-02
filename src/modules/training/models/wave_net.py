"""WaveNet model for eeg data"""
from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
# detail


class WaveNet(nn.Module):
    def __init__(self, input_channels=8, n_classes=6, kernel_size=3):
        super().__init__()
        # self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.wave_block1 = Wave_Block(input_channels, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)

        x = self.wave_block4(x)

        x_out, _ = self.rnn(x.transpose(1, 2))
        x = x_out[:, -1, :]
        x = self.fc(x)
        return x
