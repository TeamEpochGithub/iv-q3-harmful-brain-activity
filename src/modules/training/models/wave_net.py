from torch import nn
import time
import torch
import torch.nn.functional as F


# class Wave_Block(nn.Module):
#
#     def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
#         super(Wave_Block, self).__init__()
#         self.num_rates = dilation_rates
#         self.convs = nn.ModuleList()
#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#
#         self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
#         dilation_rates = [2 ** i for i in range(dilation_rates)]
#         for dilation_rate in dilation_rates:
#             self.filter_convs.append(
#                 nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
#             self.gate_convs.append(
#                 nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
#             self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))
#
#     def forward(self, x):
#         x = self.convs[0](x)
#         res = x
#         for i in range(self.num_rates):
#             x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
#             x = self.convs[i + 1](x)
#             res = res + x
#         return res
# detail
# class Wave_Block(nn.Module):
#
#     def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
#         super(Wave_Block, self).__init__()
#         self.num_rates = dilation_rates
#         self.convs = nn.ModuleList()
#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#
#         self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))
#         dilation_rates = [2 ** i for i in range(dilation_rates)]
#         for dilation_rate in dilation_rates:
#             self.filter_convs.append(
#                 nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
#             self.gate_convs.append(
#                 nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
#             self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))
#
#         for i in range(len(self.convs)):
#             nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
#             nn.init.zeros_(self.convs[i].bias)
#
#         for i in range(len(self.filter_convs)):
#             nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
#             nn.init.zeros_(self.filter_convs[i].bias)
#
#         for i in range(len(self.gate_convs)):
#             nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
#             nn.init.zeros_(self.gate_convs[i].bias)
#
#     def forward(self, x):
#         x = self.convs[0](x)
#         res = x
#         for i in range(self.num_rates):
#             tanh_out = torch.tanh(self.filter_convs[i](x))
#             sigm_out = torch.sigmoid(self.gate_convs[i](x))
#             x = tanh_out * sigm_out
#             x = self.convs[i + 1](x)
#             res = res + x
#         return res
#
#
# class Classifier(nn.Module):
#     def __init__(self, inch=3, kernel_size=3, n_classes: int = 3):
#         super().__init__()
#         self.LSTM = nn.GRU(input_size=128, hidden_size=128, num_layers=1,
#                            batch_first=True, bidirectional=True)
#
#         # self.wave_block1 = Wave_Block(inch, 16, 12, kernel_size)
#         self.wave_block2 = Wave_Block(inch, 32, 8, kernel_size)
#         self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
#         self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
#         self.fc1 = nn.Linear(256, n_classes)
#
#     def forward(self, x):
#         # x = self.wave_block1(x)
#         # curr_time = time.time()
#         x = self.wave_block2(x)
#         # block1_time = time.time()
#         x = self.wave_block3(x)
#         # block2_time = time.time()
#         x = self.wave_block4(x)
#         # block_time = time.time()
#         # print(f"Block1:{block1_time-curr_time}, Block2:{block2_time-block1_time}, Block3:{block_time-block2_time}")
#         x = x.permute(0, 2, 1)
#         x, h = self.LSTM(x)
#         x = x[:, -1, :]
#         x = self.fc1(x)
#
#         return x
class Wave_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: int, kernel_size: int = 3):
        """
        WaveNet building block.
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param dilation_rates: how many levels of dilations are used.
        :param kernel_size: size of the convolving kernel.
        """
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))

        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))

        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x)
            res = res + x
        return res


class Classifier(nn.Module):
    def __init__(self, inch=3, n_classes: int = 3):
        super().__init__()
        self.LSTM1 = nn.GRU(input_size=inch, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.GRU = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        # self.attention = Attention(input_size,4000)
        # self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')

        self.wave_block1 = Wave_Block(inch, 16, 2)
        self.dropout = nn.Dropout(0.1)
        self.wave_block2 = Wave_Block(16, 32, 1)
        # self.wave_block3 = Wave_Block(32, 64, 4)
        # self.wave_block4 = Wave_Block(64, 128, 1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        # x, _ = self.LSTM1(x.transpose(2, 1))
        # x = x.permute(0, 2, 1)

        x = self.wave_block1(x)
        x = self.wave_block2(x)
        # x = self.wave_block3(x)

        # x,_ = self.LSTM(x)
        # x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, _ = self.GRU(x)
        x = x[:, -1, :]
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.rnn(x)
        # x = self.attention(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
