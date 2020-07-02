"""
借助于已有的卷积操作实现 DWT 和 IDWT，它们的滤波器组可以是可训练参数，初始化这些滤波器组时候使用某个给定的小波
各个层中参数 trainable 默认设置为 False，表示不对滤波器组进行训练更新；更改为 True 则表示对滤波器组进行更新。
输入数据某个维度上的尺寸小于 kernel_size，会报错
在信号边界处无法精确重构
这些层暂时只支持2进制的标量小波。其他小波或超小波，如a进制小波、多小波、小波框架、曲波、脊波、条带波、小波框架等暂不适用
目前三维数据的边界延拓没有 'reflect'，因此即便使用 haar 小波也无法精确重建
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D', 'IDWT_2D', 'DWT_3D', 'IDWT_3D']
Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']

class DWT_1D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
        :param pad_type: 对输入数据的边界延拓方式，理论上使用对称小波如 bior2.2\bior3.3 等，同时对数据进行对称延拓，可以精确重构原数据，
                         但是脚本的实现有点问题，除非使用 haar 小波，否则无法精确重构，可能是由于 python 包 pywt 中的小波滤波器组的排列方式引起的
        :param wavename: 对滤波器初始化使用的小波，暂时只支持 2 进制的标量小波。
                         其他小波或超小波，如 a 进制小波、多小波、小波框架、曲波、脊波、条带波、小波框架等暂不适用；
                         对于 2D/3D 数据，相应的滤波器是由 1D 滤波器组进行张量相乘得到的，对应的小波称为张量小波或可分离小波，若要使用不可分离小波，则要重建脚本
        :param stride: 采样步长，脚本设置这个值必须为2，非要设置为其他数值也是可以运行的（此时需屏蔽脚本中的 assert self.stride == 2），但是不满足小波理论；
                        若是用任意进制的小波，如3进制小波，可相应调整这个参数，但此时有更多的滤波器组，会相应分解出更多高频分量，对应的还要更新脚本内容
        :param in_channels: 输入数据的通道数
        :param out_channels: 输出数据的通道数，默认与输入数据通道数相同
        :param groups: 对通道这一维度的分组数目，这个值需要能被 in_channels 整除，
                        默认值与输入数据的通道数相同，即为 in_channels；一般的卷积操作这里默认值为 1
        :param kernel_size: 卷积核尺寸，这个参数与参数 wavename 有一定的冲突，即该参数值必须大于初始化小波滤波器长度；
                            该参数的默认值是等于初始化所用小波滤波器长度
                            若训练过程中不对滤波器组进行学习更新，即参数 trainable 设置为 False，则建议参数 kernel_size 选用默认值，因为此时除了运算量的提升，并不能带来任何增益
                            若参数 trainable 设置为 True，参数 kernel_size 应大于等于初始化所用小波的滤波器长度，此时有可能训练得到更适用于当前数据分布的滤波器组
                            个人不建议 kernel_size 的值设置的比初始化小波滤波器长度大的很多，个人建议这个超出值不要大于 3
        :param trainable: 标记是否在训练过程中更新滤波器组参数；
                          若这个参数设置为 True，且同时 groups 设置为 1 ，那么：
                                DWT层等价于多个 stride = 2 的卷积层，只是对卷积核的大小以及初始化方式不同
                                IDWT层等价于多个 stride = 2 的反卷积层操作后相加，同样卷积核的大小以及初始化方式不同

                当 out_channels 和 groups 都采用默认值时，对应的是对输入数据逐通道进行小波变换
                当 groups 取值为 1 时候，与一般的卷积操作有相似，可理解为融合数据在不同通道的相同频段内的信息
                与一般的卷积层一样，理论上这些层可以处理任意尺寸的数据。
                但是，如果输入数据某个维度上尺寸小于滤波器组长度的1/2，在重构过程中对数据延拓时会报错
                另外，我们建议输入数据各个维度上的尺寸是偶数值。

                其他各层需要说明的事项与此基本相同，不再说明。
        """
        super(DWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0, '参数 groups 的应能被 in_channels 整除'
        self.stride = stride
        assert self.stride == 2, '目前版本，stride 只能等于 2'
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low  = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)

        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 3
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv1d(input, self.filter_low, stride = self.stride, groups = self.groups), \
               F.conv1d(input, self.filter_high, stride = self.stride, groups = self.groups)


class IDWT_1D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
            参照 DWT_1D 中的说明
            理论上，使用简单上采样和卷积实现的 IDWT 要比矩阵法计算量小、速度快，
            然而由于 Pytorch 中没有实现简单上采样，在实现 IDWT 只能用与 [1,0] 做反卷积 Deconvolution 来实现简单上采样
            这使得该方法比矩阵法实现 IDWT 速度慢非常多。
        """
        super(IDWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1.0
        up_filter = up_filter[None, None, :].repeat((self.in_channels, 1, 1))
        self.register_buffer('up_filter', up_filter)
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
            self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, L, H):
        assert len(L.size()) == len(H.size()) == 3
        assert L.size()[0] == H.size()[0]
        assert L.size()[1] == H.size()[1] == self.in_channels
        L = F.pad(F.conv_transpose1d(L, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        H = F.pad(F.conv_transpose1d(H, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        return F.conv1d(L, self.filter_low, stride = 1, groups = self.groups) + \
               F.conv1d(H, self.filter_high, stride = 1, groups = self.groups)


class DWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
            参照 DWT_1D 中的说明
        """
        super(DWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_ll = self.filt_low[:, None] * self.filt_low[None,:]
        self.filter_lh = self.filt_low[:, None] * self.filt_high[None,:]
        self.filter_hl = self.filt_high[:, None] * self.filt_low[None,:]
        self.filter_hh = self.filt_high[:, None] * self.filt_high[None,:]
        self.filter_ll = self.filter_ll[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_lh = self.filter_lh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hl = self.filter_hl[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hh = self.filter_hh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        if torch.cuda.is_available():
            self.filter_ll = self.filter_ll.cuda()
            self.filter_lh = self.filter_lh.cuda()
            self.filter_hl = self.filter_hl.cuda()
            self.filter_hh = self.filter_hh.cuda()
        if self.trainable:
            self.filter_ll = nn.Parameter(self.filter_ll)
            self.filter_lh = nn.Parameter(self.filter_lh)
            self.filter_hl = nn.Parameter(self.filter_hl)
            self.filter_hh = nn.Parameter(self.filter_hh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(input, self.filter_ll, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_lh, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_hl, stride = self.stride, groups = self.groups),\
               F.conv2d(input, self.filter_hh, stride = self.stride, groups = self.groups)


class IDWT_2D(nn.Module):
    def __init__(self, pad_type = 'reflect', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups = None,
                 kernel_size = None, trainable = False):
        """
            参照 DWT_1D 中的说明
            理论上，使用简单上采样和卷积实现的 IDWT 要比矩阵法计算量小、速度快，
            然而由于 Pytorch 中没有实现简单上采样，在实现 IDWT 只能用与 [[1,0],[0,0]] 做反卷积 Deconvolution 来实现简单上采样
            这使得该方法比矩阵法实现 IDWT 速度慢非常多。
            目前，在论文 https://arxiv.org/abs/2005.14461 中，构建 WaveSNet 事实上仍使用 DWT/IDWT 的矩阵法实现
        """
        super(IDWT_2D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_ll = self.filt_low[:, None] * self.filt_low[None,:]
        self.filter_lh = self.filt_low[:, None] * self.filt_high[None,:]
        self.filter_hl = self.filt_high[:, None] * self.filt_low[None,:]
        self.filter_hh = self.filt_high[:, None] * self.filt_high[None,:]
        self.filter_ll = self.filter_ll[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_lh = self.filter_lh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hl = self.filter_hl[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        self.filter_hh = self.filter_hh[None, None, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1))
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1
        up_filter = up_filter[:, None] * up_filter[None,:]
        up_filter = up_filter[None, None, :, :].repeat(self.out_channels, 1, 1, 1)
        self.register_buffer('up_filter', up_filter)
        if torch.cuda.is_available():
            self.filter_ll = self.filter_ll.cuda()
            self.filter_lh = self.filter_lh.cuda()
            self.filter_hl = self.filter_hl.cuda()
            self.filter_hh = self.filter_hh.cuda()
            #self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_ll = nn.Parameter(self.filter_ll)
            self.filter_lh = nn.Parameter(self.filter_lh)
            self.filter_hl = nn.Parameter(self.filter_hl)
            self.filter_hh = nn.Parameter(self.filter_hh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        assert LL.size()[0] == LH.size()[0] == HL.size()[0] == HH.size()[0]
        assert LL.size()[1] == LH.size()[1] == HL.size()[1] == HH.size()[1] == self.in_channels
        LL = F.conv_transpose2d(LL, self.up_filter, stride = self.stride, groups = self.in_channels)
        LH = F.conv_transpose2d(LH, self.up_filter, stride = self.stride, groups = self.in_channels)
        HL = F.conv_transpose2d(HL, self.up_filter, stride = self.stride, groups = self.in_channels)
        HH = F.conv_transpose2d(HH, self.up_filter, stride = self.stride, groups = self.in_channels)
        LL = F.pad(LL, pad = self.pad_sizes, mode = self.pad_type)
        LH = F.pad(LH, pad = self.pad_sizes, mode = self.pad_type)
        HL = F.pad(HL, pad = self.pad_sizes, mode = self.pad_type)
        HH = F.pad(HH, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv2d(LL, self.filter_ll, stride = 1, groups = self.groups) + \
               F.conv2d(LH, self.filter_lh, stride = 1, groups = self.groups) + \
               F.conv2d(HL, self.filter_hl, stride = 1, groups = self.groups) + \
               F.conv2d(HH, self.filter_hh, stride = 1, groups = self.groups)


class DWT_3D(nn.Module):
    def __init__(self, pad_type = 'replicate', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups=None,
                 kernel_size = None, trainable = False):
        """
            参照 DWT_1D 中的说明
        """
        super(DWT_3D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_lll = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_llh = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_lhl = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_lhh = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_hll = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_hlh = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_hhl = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_hhh = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_lll = self.filter_lll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_llh = self.filter_llh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhl = self.filter_lhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhh = self.filter_lhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hll = self.filter_hll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hlh = self.filter_hlh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhl = self.filter_hhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhh = self.filter_hhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        if torch.cuda.is_available():
            self.filter_lll = nn.Parameter(self.filter_lll).cuda()
            self.filter_llh = nn.Parameter(self.filter_llh).cuda()
            self.filter_lhl = nn.Parameter(self.filter_lhl).cuda()
            self.filter_lhh = nn.Parameter(self.filter_lhh).cuda()
            self.filter_hll = nn.Parameter(self.filter_hll).cuda()
            self.filter_hlh = nn.Parameter(self.filter_hlh).cuda()
            self.filter_hhl = nn.Parameter(self.filter_hhl).cuda()
            self.filter_hhh = nn.Parameter(self.filter_hhh).cuda()
        if self.trainable:
            self.filter_lll = nn.Parameter(self.filter_lll)
            self.filter_llh = nn.Parameter(self.filter_llh)
            self.filter_lhl = nn.Parameter(self.filter_lhl)
            self.filter_lhh = nn.Parameter(self.filter_lhh)
            self.filter_hll = nn.Parameter(self.filter_hll)
            self.filter_hlh = nn.Parameter(self.filter_hlh)
            self.filter_hhl = nn.Parameter(self.filter_hhl)
            self.filter_hhh = nn.Parameter(self.filter_hhh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 5
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv3d(input, self.filter_lll, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_llh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_lhl, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_lhh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hll, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hlh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hhl, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hhh, stride = self.stride, groups = self.groups)


class IDWT_3D(nn.Module):
    def __init__(self, pad_type = 'replicate', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups=None,
                 kernel_size = None, trainable=False):
        """
            参照 DWT_1D 中的说明
            理论上，使用简单上采样和卷积实现的 IDWT 要比矩阵法计算量小、速度快，
            然而由于 Pytorch 中没有实现简单上采样，在实现 IDWT 只能用与 [[[1,0],[0,0]], [[0,0],[0,0]]] 做反卷积 Deconvolution 来实现简单上采样
            这使得该方法比矩阵法实现 IDWT 速度慢非常多。
        """
        super(IDWT_3D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_lll = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_llh = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_lhl = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_lhh = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_hll = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_hlh = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_hhl = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_hhh = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_lll = self.filter_lll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_llh = self.filter_llh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhl = self.filter_lhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhh = self.filter_lhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hll = self.filter_hll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hlh = self.filter_hlh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhl = self.filter_hhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhh = self.filter_hhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1
        up_filter = up_filter[:, None, None] * up_filter[None,:, None] * up_filter[None, None, :]
        up_filter = up_filter[None, None, :, :, :].repeat(self.out_channels, 1, 1, 1, 1)
        self.register_buffer('up_filter', up_filter)
        if torch.cuda.is_available():
            self.filter_lll = nn.Parameter(self.filter_lll).cuda()
            self.filter_llh = nn.Parameter(self.filter_llh).cuda()
            self.filter_lhl = nn.Parameter(self.filter_lhl).cuda()
            self.filter_lhh = nn.Parameter(self.filter_lhh).cuda()
            self.filter_hll = nn.Parameter(self.filter_hll).cuda()
            self.filter_hlh = nn.Parameter(self.filter_hlh).cuda()
            self.filter_hhl = nn.Parameter(self.filter_hhl).cuda()
            self.filter_hhh = nn.Parameter(self.filter_hhh).cuda()
            self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_lll = nn.Parameter(self.filter_lll)
            self.filter_llh = nn.Parameter(self.filter_llh)
            self.filter_lhl = nn.Parameter(self.filter_lhl)
            self.filter_lhh = nn.Parameter(self.filter_lhh)
            self.filter_hll = nn.Parameter(self.filter_hll)
            self.filter_hlh = nn.Parameter(self.filter_hlh)
            self.filter_hhl = nn.Parameter(self.filter_hhl)
            self.filter_hhh = nn.Parameter(self.filter_hhh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        assert LLL.size()[0] == LLH.size()[0] == LHL.size()[0] == LHH.size()[0] == HLL.size()[0] == HLH.size()[0] == HHL.size()[0] == HHH.size()[0]
        assert LLL.size()[1] == LLH.size()[1] == LHL.size()[1] == LHH.size()[1] == HLL.size()[1] == HLH.size()[1] == HHL.size()[1] == HHH.size()[1] == self.in_channels
        LLL = F.pad(F.conv_transpose3d(LLL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LLH = F.pad(F.conv_transpose3d(LLH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LHL = F.pad(F.conv_transpose3d(LHL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LHH = F.pad(F.conv_transpose3d(LHH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HLL = F.pad(F.conv_transpose3d(HLL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HLH = F.pad(F.conv_transpose3d(HLH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HHL = F.pad(F.conv_transpose3d(HHL, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HHH = F.pad(F.conv_transpose3d(HHH, self.up_filter, stride = self.stride, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        return F.conv3d(LLL, self.filter_lll, stride = 1, groups = self.groups) + \
               F.conv3d(LLH, self.filter_llh, stride = 1, groups = self.groups) + \
               F.conv3d(LHL, self.filter_lhl, stride = 1, groups = self.groups) + \
               F.conv3d(LHH, self.filter_lhh, stride = 1, groups = self.groups) + \
               F.conv3d(HLL, self.filter_hll, stride = 1, groups = self.groups) + \
               F.conv3d(HLH, self.filter_hlh, stride = 1, groups = self.groups) + \
               F.conv3d(HHL, self.filter_hhl, stride = 1, groups = self.groups) + \
               F.conv3d(HHH, self.filter_hhh, stride = 1, groups = self.groups)


if __name__ == '__main__':
    """
    import numpy as np
    vector = np.array(range(3*2*8)).reshape((3,2,8))
    print(vector)
    wavename = 'haar'
    vector = torch.tensor(vector).float()
    m0 = DWT_1D(wavename = wavename, in_channels = 2, kernel_size = 12, trainable = True)
    L, H = m0(vector)
    print('L size is {}'.format(L.size()))
    print('L is {}'.format(L))
    print('H size is {}'.format(H.size()))
    print('H is {}'.format(H))
    m1 = IDWT_1D(wavename = wavename, in_channels = 2, kernel_size = 12, trainable = True)
    vector_re = m1(L, H)
    print(vector_re)
    print(vector - vector_re)
    print(torch.max(vector - vector_re), torch.min(vector - vector_re))
    """
    """
    import cv2
    import numpy as np
    from DWT_IDWT_layer import DWT_2D as DWT_2D_old
    from DWT_IDWT_layer import IDWT_2D as IDWT_2D_old

    imagename = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(imagename)
    cv2.imshow('image', image)
    image_tensor = torch.tensor(image).float()
    image_tensor.transpose_(dim0 = -1, dim1 = 0).transpose_(dim0 = -1, dim1 = 1)
    image_tensor.unsqueeze_(dim = 0)
    print(image_tensor.size())
    wavename = 'haar'
    m = DWT_2D(wavename = wavename, in_channels = 3, kernel_size = 2, trainable = True)
    m_o = DWT_2D_old(wavename = wavename)
    LL, LH, HL, HH = m(image_tensor)
    LL_o, LH_o, HL_o, HH_o = m_o(image_tensor)
    print('LL == > {}, {}'.format(torch.max(LL - LL_o), torch.min(LL - LL_o)))
    print('LH == > {}, {}'.format(torch.max(LH - LH_o), torch.min(LH - LH_o)))
    print('HL == > {}, {}'.format(torch.max(HL - HL_o), torch.min(HL - HL_o)))
    print('HH == > {}, {}'.format(torch.max(HH - HH_o), torch.min(HH - HH_o)))
    m1 = IDWT_2D(wavename = wavename, in_channels = 3, kernel_size = 2, trainable = True)
    m1_o = IDWT_2D_old(wavename = wavename)
    image_re = m1(LL, LH, HL, HH)
    image_re_o = m1_o(LL_o, LH_o, HL_o, HH_o)
    print('image_re size is {}'.format(image_re.size()))
    print('LL size is {}'.format(LL.size()))
    # print(torch.max(torch.abs(image_tensor - image_re)), torch.min(torch.abs(image_tensor - image_re)))
    image_re.squeeze_().transpose_(dim0 = -1, dim1 = 1).transpose_(dim0 = -1, dim1 = 0)
    image_re_o.squeeze_().transpose_(dim0 = -1, dim1 = 1).transpose_(dim0 = -1, dim1 = 0)
    image_re = np.array(image_re.data)
    image_re_o = np.array(image_re_o)
    cv2.imshow('image_re', image_re / np.max(image_re))

    print(np.max(image - image_re), np.min(image - image_re))
    gap = 0
    gap = gap
    gap_ = -gap if gap != 0 else None
    print(np.max((image - image_re)[gap:gap_, gap:gap_, 2]), np.min((image - image_re)[gap:gap_, gap:gap_, :]))
    cv2.imshow('---', (image - image_re) / np.max(image - image_re))
    cv2.imshow('------', (image - image_re_o) / np.max(image - image_re_o))
    print((image - image_re)[64, 0:20, 0])
    print((image - image_re)[64, -20:, 0])
    print((image_re_o - image_re)[64, 0:20, 0])
    print((image_re_o - image_re)[64, -20:, 0])
    cv2.waitKey(0)
    """
    #"""
    wavename = 'haar'
    vector = torch.ones((1,1,8,8,8))
    m0 = DWT_3D(wavename = wavename, in_channels = 1, kernel_size = 4, trainable = True)
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = m0(vector)
    print('LLL size is {}'.format(LLL.size()))
    m1 = IDWT_3D(wavename = wavename, in_channels = 1, kernel_size = 4, trainable = True)
    vector_re = m1(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
    print(vector - vector_re)
    #"""

