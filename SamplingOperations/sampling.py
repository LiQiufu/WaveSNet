import torch
from torch import nn
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D

class My_DownSampling_SC(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = (1,1), stride = 2, padding = (0,0)):
        super(My_DownSampling_SC, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, input):
        return self.conv(input), input


class My_DownSampling_MP(nn.Module):
    def __init__(self, stride = 2, kernel_size = 2):
        super(My_DownSampling_MP, self).__init__()
        self.maxp = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, return_indices = False)

    def forward(self, input):
        return self.maxp(input), input


class My_UpSampling_SC(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = (1,1), stride = 2, padding = (0,0)):
        super(My_UpSampling_SC, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, input, feature_map):
        return torch.cat((self.conv(input), feature_map), dim = 1)


class My_DownSampling_DWT(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(My_DownSampling_DWT, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return LL, LH, HL, HH, input


class My_UpSampling_IDWT(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(My_UpSampling_IDWT, self).__init__()
        self.idwt = IDWT_2D(wavename = wavename)

    def forward(self, LL, LH, HL, HH, feature_map):
        return torch.cat((self.idwt(LL, LH, HL, HH), feature_map), dim = 1)