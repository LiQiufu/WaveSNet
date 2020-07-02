from WDeepLabV3Plus.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, wavename = 'haar'):
    if backbone == 'resnet' or backbone == 'resnet101':
        return resnet.ResNet101(output_stride, BatchNorm, wavename = wavename)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
