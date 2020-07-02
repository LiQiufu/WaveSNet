"""
segnet_vgg16_bn and wsegnet_vgg16_bn are the SegNet and WSegNet used in the paper:
https://arxiv.org/abs/2005.14461
"""

import torch.nn as nn
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from SegNet_UNet import *
import torch
__all__ = [
    'SegNet_VGG', 'segnet_vgg11', 'segnet_vgg11_bn', 'segnet_vgg13', 'segnet_vgg13_bn', 'segnet_vgg16', 'segnet_vgg16_bn',
    'segnet_vgg19_bn', 'segnet_vgg19', 'wsegnet_vgg11', 'wsegnet_vgg11_bn', 'wsegnet_vgg13', 'wsegnet_vgg13_bn', 'wsegnet_vgg16', 'wsegnet_vgg16_bn',
    'wsegnet_vgg19_bn', 'wsegnet_vgg19',
]

class SegNet_VGG(nn.Module):
    def __init__(self, features, num_classes = 21, init_weights = True, wavename = None):
        super(SegNet_VGG, self).__init__()
        self.features = features[0]
        self.decoders = features[1]
        self.classifier_seg = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            #nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size = 1, padding = 0),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        xx = self.features(x)
        x, [(indices_1,), (indices_2,), (indices_3,), (indices_4,), (indices_5,)] = xx
        x = self.decoders(x, indices_5, indices_4, indices_3, indices_2, indices_1)
        x = self.classifier_seg(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return 'SegNet_VGG'


class WSegNet_VGG(nn.Module):
    def __init__(self, features, num_classes = 21, init_weights = True, wavename = None):
        super(WSegNet_VGG, self).__init__()
        self.features = features[0]
        self.decoders = features[1]
        self.classifier_seg = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            #nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size = 1, padding = 0),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        xx = self.features(x)
        x, [(LH1,HL1,HH1), (LH2,HL2,HH2,), (LH3,HL3,HH3,), (LH4,HL4,HH4,), (LH5,HL5,HH5,)] = xx
        x = self.decoders(x, LH5,HL5,HH5, LH4,HL4,HH4, LH3,HL3,HH3, LH2,HL2,HH2, LH1,HL1,HH1)
        x = self.classifier_seg(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return 'WSegNet_VGG'


def make_layers(cfg, batch_norm = False):
    encoder = []
    in_channels = 3
    for v in cfg:
        if v != 'M':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                encoder += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                encoder += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            encoder += [nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices = True)]
    encoder = My_Sequential(*encoder)

    decoder = []
    cfg.reverse()
    out_channels_final = 64
    for index, v in enumerate(cfg):
        if index != len(cfg) - 1:
            out_channels = cfg[index + 1]
        else:
            out_channels = out_channels_final
        if out_channels == 'M':
            out_channels = cfg[index + 2]
        if v == 'M':
            decoder += [nn.MaxUnpool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(v, out_channels, kernel_size = 3, padding = 1)
            if batch_norm:
                decoder += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True)]
            else:
                decoder += [conv2d, nn.ReLU(inplace = True)]
    decoder = My_Sequential_re(*decoder)
    return encoder, decoder


def make_w_layers(cfg, batch_norm = False, wavename = 'haar'):
    encoder = []
    in_channels = 3
    for v in cfg:
        if v != 'M':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                encoder += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                encoder += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            encoder += [DWT_2D(wavename = wavename)]
    encoder = My_Sequential(*encoder)

    decoder = []
    cfg.reverse()
    out_channels_final = 64
    for index, v in enumerate(cfg):
        if index != len(cfg) - 1:
            out_channels = cfg[index + 1]
        else:
            out_channels = out_channels_final
        if out_channels == 'M':
            out_channels = cfg[index + 2]
        if v == 'M':
            decoder += [IDWT_2D(wavename = wavename)]
        else:
            conv2d = nn.Conv2d(v, out_channels, kernel_size = 3, padding = 1)
            if batch_norm:
                decoder += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True)]
            else:
                decoder += [conv2d, nn.ReLU(inplace = True)]
    decoder = My_Sequential_re(*decoder)
    return encoder, decoder


cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],   # 11 layers
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],   # 13 layers
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],   # 16 layers out_channels for encoder, input_channels for decoder
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],   # 19 layers
}

def segnet_vgg11(pretrained = False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['A']), **kwargs)
    return model


def segnet_vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['A'], batch_norm = True), **kwargs)
    return model


def segnet_vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['B']), **kwargs)
    return model


def segnet_vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def segnet_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['D']), **kwargs)
    return model


def segnet_vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def segnet_vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['E']), **kwargs)
    return model


def segnet_vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SegNet_VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

"""================================================================================="""

def wsegnet_vgg11(pretrained = False, wavename = 'haar', **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['A'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg11_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['A'], batch_norm = True, wavename = wavename), **kwargs)
    return model


def wsegnet_vgg13(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['B'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg13_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['B'], batch_norm=True, wavename = wavename), **kwargs)
    return model


def wsegnet_vgg16(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['D'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg16_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['D'], batch_norm=True, wavename = wavename), **kwargs)
    return model


def wsegnet_vgg19(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['E'], wavename = wavename), **kwargs)
    return model


def wsegnet_vgg19_bn(pretrained=False, wavename = 'haar', **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = WSegNet_VGG(make_w_layers(cfg['E'], batch_norm=True, wavename = wavename), **kwargs)
    return model


if __name__ == '__main__':
    x = torch.rand(size = (1,3,448,448))
    net = wsegnet_vgg16_bn()
    y = net(x)