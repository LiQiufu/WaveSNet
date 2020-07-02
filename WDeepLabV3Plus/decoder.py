import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from WDeepLabV3Plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from DWT_IDWT.DWT_IDWT_layer import IDWT_2D

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, wavename = 'haar', p_dropout = (0.5, 0.25)):
        """
        :param num_classes:
        :param backbone:
        :param BatchNorm:
        :param wavename:
        :param p_dropout: (0.5, 0.25) for pascal voc; (0, 0.1) for cityscapes with haar,
        """
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn' or backbone == 'resnet101':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.idwt = IDWT_2D(wavename = wavename)
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.conv_lh3 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv_hl3 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv_hh3 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.last_conv_0 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         BatchNorm(256),
                                         nn.ReLU())
        self.last_conv_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         BatchNorm(256),
                                         nn.ReLU(),
                                         nn.Dropout(p_dropout[0]),
                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         BatchNorm(256),
                                         nn.ReLU(),
                                         nn.Dropout2d(p_dropout[1]),
                                         nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        [(LH2, HL2, HH2), (LH3, HL3, HH3)] = low_level_feat
        LH3 = self.conv_lh3(LH3)
        HL3 = self.conv_hl3(HL3)
        HH3 = self.conv_hh3(HH3)
        x = self.idwt(x, LH3, HL3, HH3)
        #_,_,h,w = x.size()
        #x = F.interpolate(x, size = (2*h, 2*w), mode = 'bilinear', align_corners = True)
        x = self.last_conv_0(x)
        x = self.idwt(x, LH2, HL2, HH2)
        #_,_,h,w = x.size()
        #x = F.interpolate(x, size = (2*h, 2*w), mode = 'bilinear', align_corners = True)
        x = self.last_conv_1(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, wavename = 'haar', p_dropout = (0.5, 0.1)):
    return Decoder(num_classes, backbone, BatchNorm, wavename = wavename, p_dropout = p_dropout)