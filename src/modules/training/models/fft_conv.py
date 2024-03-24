"""Model that uses convolutions on FFT results."""

import torch
from torch import nn



class FFTConv(nn.Module):
    def __init__(self, in_channels: int = 20, intermediate_channels: int = 32, out_channels: int = 6):
        super().__init__()
        self.conv_list1 = nn.ModuleList(
            [nn.Conv1d(in_channels, intermediate_channels, kernel_size=100, padding=0) for _ in range(100)])
        self.conv_list2 = nn.ModuleList(
            [nn.Conv1d(intermediate_channels, out_channels, kernel_size=100, padding=0) for _ in range(1)])
        self.relu = nn.ReLU()
        self.intermediate_channels = intermediate_channels

    def forward(self, x):
        # Given x takethe fft of the input
        x = torch.fft.fft(x, dim=-1)
        x_abs = torch.log(torch.abs(x))
        x2 = torch.zeros(x.shape[0], self.intermediate_channels, 100).to('cuda')
        out = torch.zeros(x.shape[0], 6).to('cuda')
        # Apply the convolutions
        for i, n in enumerate(range(0,10000, 100)):
            x2[:,:,i] = self.conv_list1[i](x_abs[:,:,n:n+100]).squeeze(-1)
            x2[:,:,i] = self.relu(x2[:,:,i])
        for i in range(1):
            out = self.conv_list2[i](x2).squeeze(-1)
            out = self.relu(out)
        
        return out
