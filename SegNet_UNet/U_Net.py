"""
unet_vgg16_bn is the U-Net used in the paper:
https://arxiv.org/abs/2005.14461
"""
from SamplingOperations.sampling import *
from SegNet_UNet import *
import torch
__all__ = [
    'U_Net_VGG',
]


class U_Net_VGG(nn.Module):
    def __init__(self, features, num_classes = 21, init_weights = True, wavename = None):
        super(U_Net_VGG, self).__init__()
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
        x, [(feature_map_1,), (feature_map_2,), (feature_map_3,), (feature_map_4,), (feature_map_5,)] = xx
        x = self.decoders(x, feature_map_5, feature_map_4, feature_map_3, feature_map_2, feature_map_1)
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
        return 'U_Net_VGG'


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
            encoder += [My_DownSampling_MP(kernel_size = 2, stride = 2)]
    encoder = My_Sequential(*encoder)

    decoder = []
    cfg.reverse()
    out_channels_final = 64
    in_channels = 512
    for index, v in enumerate(cfg):
        if index != len(cfg) - 1:
            out_channels = cfg[index + 1]
        else:
            out_channels = out_channels_final
        if out_channels == 'M':
            out_channels = cfg[index + 2]
        if v == 'M':
            decoder += [My_UpSampling_SC(in_channel = in_channels, out_channel = out_channels, kernel_size = 2, stride = 2)]
        else:
            if cfg[index - 1] == 'M':
                v = 2 * v
            conv2d = nn.Conv2d(v, out_channels, kernel_size = 3, padding = 1)
            if batch_norm:
                decoder += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True)]
            else:
                decoder += [conv2d, nn.ReLU(inplace = True)]
        in_channels = out_channels
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
            encoder += [My_DownSampling_DWT(wavename = wavename)]
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
            decoder += [My_UpSampling_IDWT(wavename = wavename)]
        else:
            if cfg[index - 1] == 'M':
                v = 2 * v
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

def unet_vgg11(pretrained = False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['A']), **kwargs)
    return model


def unet_vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['A'], batch_norm = True), **kwargs)
    return model


def unet_vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['B']), **kwargs)
    return model


def unet_vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def unet_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['D']), **kwargs)
    return model


def unet_vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def unet_vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['E']), **kwargs)
    return model


def unet_vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = U_Net_VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model



if __name__ == '__main__':
    x = torch.rand(size = (1,3,448,448))
    net = unet_vgg16_bn()
    y = net(x)