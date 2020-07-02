# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
"""
自定义pytorch层，实现一维、二维、三维张量的DWT和IDWT，未考虑边界延拓
只有当图像行列数都是偶数，且重构滤波器组低频分量长度为2时，才能精确重构，否则在边界处有误差。
与论文中公式有一处差异，主要是在二维情况下：论文中是LH = HXL，程序中实现是 LH = LXH
"""
import numpy as np
#import cv2
import math, pywt
from torch.nn import Module
from DWT_IDWT.DWT_IDWT_Functions import *

__all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D', 'IDWT_2D', 'DWT_3D', 'IDWT_3D', 'DWT_2D_tiny']
class DWT_1D(Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, input):
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        #assert self.input_height > self.band_length
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  L -- (N, C, L/2)
            H -- (N, C, L/2)
    output: (N, C, L)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重建所需低频滤波器组
        :param band_high: 小波重建所需高频滤波器组
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.tensor(matrix_h).cuda()
            self.matrix_high = torch.tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.tensor(matrix_h)
            self.matrix_high = torch.tensor(matrix_g)

    def forward(self, L, H):
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        #assert self.input_height > self.band_length
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


class DWT_2D(Module):
    """
    input: (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
              LH: (N, C, H/2, W/2)
              HL: (N, C, H/2, W/2)
              HH: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_2D_tiny(Module):
    """
    input: (N, C, H, W)
    output -- LL: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class IDWT_2D(Module):
    """
    input -- LL: (N, C, H/2, W/2)
             LH: (N, C, H/2, W/2)
             HL: (N, C, H/2, W/2)
             HH: (N, C, H/2, W/2)
    output: (N, C, H, W)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重建所需低频滤波器组
        :param band_high: 小波重建所需高频滤波器组
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_3D(Module):
    """
    input: (N, C, D, H, W)
    output: -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波分解所用低频滤波器组
        :param band_high: 小波分解所用高频滤波器组
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_low_2 = torch.tensor(matrix_h_2)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)
            self.matrix_high_2 = torch.tensor(matrix_g_2)

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        return DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


class IDWT_3D(Module):
    """
    input:  -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    output: (N, C, D, H, W)
    """
    def __init__(self, wavename):
        """
        :param band_low: 小波重构所用低频滤波器组
        :param band_high: 小波重构所用高频滤波器组
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        :return:
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.tensor(matrix_h_0)
            self.matrix_low_1 = torch.tensor(matrix_h_1)
            self.matrix_low_2 = torch.tensor(matrix_h_2)
            self.matrix_high_0 = torch.tensor(matrix_g_0)
            self.matrix_high_1 = torch.tensor(matrix_g_1)
            self.matrix_high_2 = torch.tensor(matrix_g_2)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL.size()[-3] + HHH.size()[-3]
        self.input_height = LLL.size()[-2] + HHH.size()[-2]
        self.input_width = LLL.size()[-1] + HHH.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                     self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


if __name__ == '__main__':
    import pywt, cv2
    from datetime import datetime

    wavelet = pywt.Wavelet('bior1.1')
    h = wavelet.rec_lo
    g = wavelet.rec_hi
    h_ = wavelet.dec_lo
    g_ = wavelet.dec_hi
    h_.reverse()
    g_.reverse()

    #"""
    image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    height, width, channel = image.shape
    #image = image.reshape((1,height,width))
    t0 = datetime.now()
    for index in range(1):
        m0 = DWT_2D(wavename = 'haar')

        m1 = IDWT_2D(wavename = 'haar')
    print(isinstance(m1, IDWT_2D))
    t1 = datetime.now()
    #"""
    """
    image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    image_3d = np.concatenate((image, image, image, image, image, image), axis = 2)
    print(image_3d.shape)
    image_tensor = torch.Tensor(image_3d)
    #image_tensor = image_tensor.transpose(dim0 = 2, dim1 = 1)
    #image_tensor = image_tensor.transpose(dim0 = 1, dim1 = 0)
    image_tensor.unsqueeze_(dim = 0)
    image_tensor.unsqueeze_(dim = 0)
    t0 = datetime.now()
    for index in range(10):
        m0 = DWT_3D(wavename = 'haar')
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_low_2 = torch.Tensor(m0.matrix_low_2)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)
        matrix_high_2 = torch.Tensor(m0.matrix_high_2)

        #image_tensor.requires_grad = True
        #input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #                                matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(DWTFunction_3D.apply, input)
        #print('testing dwt3d -- {}'.format(test))
        #LLL.requires_grad = True
        #input = (LLL.double(), LLH.double(), LHL.double(), LHH.double(),
        #         HLL.double(), HLH.double(), HHL.double(), HHH.double(),
        #         matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #         matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(IDWTFunction_3D.apply, input)
        #print('testing idwt3d -- {}'.format(test))

        m1 = IDWT_3D(wavename = 'haar')
        image_re = m1(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH)
    t1 = datetime.now()
    image_re.squeeze_(dim = 0)
    image_re.squeeze_(dim = 0)
    #image_re.transpose_(0,1)
    #image_re.transpose_(1,2)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[:,:,0:3]
    print(np.max(image_zero), np.min(image_zero))
    #print(image_zero[:,8,0])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[:,:,0:3]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)
    """

    """
    import matplotlib.pyplot as plt
    import numpy as np
    vector_np = np.array(list(range(1280)))#.reshape((128,1))

    print(vector_np.shape)
    t0 = datetime.now()
    for index in range(100):
        vector = torch.Tensor(vector_np)
        vector.unsqueeze_(dim = 0)
        vector.unsqueeze_(dim = 0)
        m0 = DWT_1D(band_low = h, band_high = g)
        L, H = m0(vector)

        #matrix_low = torch.Tensor(m0.matrix_low)
        #matrix_high = torch.Tensor(m0.matrix_high)
        #vector.requires_grad = True
        #input = (vector.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(DWTFunction_1D.apply, input)
        #print('testing 1D-DWT: {}'.format(test))
        #print(L.requires_grad)
        #print(H.requires_grad)
        #L.requires_grad = True
        #H.requires_grad = True
        #input = (L.double(), H.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(IDWTFunction_1D.apply, input)
        #print('testing 1D-IDWT: {}'.format(test))

        m1 = IDWT_1D(band_low = h_, band_high = g_)
        vector_re = m1(L, H)
    t1 = datetime.now()
    vector_re_np = vector_re.detach().numpy()
    print('image_re shape: {}'.format(vector_re_np.shape))

    vector_zero = vector_np - vector_re_np.reshape(vector_np.shape)
    print(np.max(vector_zero), np.min(vector_zero))
    print(vector_zero[:8])
    print('taking {} secondes'.format(t1 - t0))
    """
