"""
这个脚本构建一个数据扰动层，它的功能是给数据增加微小扰动
"""
import numpy as np
import cv2
import math
import torch
from random import random
from torch.nn import Module
from torch.autograd import Function
from datetime import datetime

class Perturbation(Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, p = 0.5, amplitude = 0.5):
        """
        :param p: 对每个数据样本进行扰动的概率，取值范围 【0,1】
        :param amplitude: 百分比，对每个数据样本进行扰动的幅度最大比例，建议取值范围： 【-1,1】
        """
        super(Perturbation, self).__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.amplitude = amplitude

    def get_random_data(self, shape):
        noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
        data_random = torch.rand(shape, out = noise)
        data_random = (data_random - 0.5) * 2 * self.amplitude
        N = shape[0]
        lines = torch.rand(N)
        lines = lines.le(self.p).float()
        if torch.cuda.is_available():
            lines = lines.cuda()
        data_random = data_random.transpose(dim0 = 0, dim1 = -1)
        data_random = lines * data_random
        data_random = data_random.transpose(dim0 = 0, dim1 = -1)
        return data_random + 1

    def forward(self, input = None):
        if not self.training:
            return input
        shape = input.size()
        assert len(shape) >= 2 and len(shape) <= 5
        data_random = self.get_random_data(shape)
        return PerturbationFunction.apply(input, data_random)


class Perturbation_o(Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, p = 0.5, amplitude = 0.5):
        """
        :param p: 对每个数据样本进行扰动的概率，取值范围 【0,1】
        :param amplitude: 百分比，对每个数据样本进行扰动的幅度最大比例，建议取值范围： 【-1,1】
        """
        super(Perturbation_o, self).__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.amplitude = amplitude

    def get_random_data(self, shape):
        if torch.cuda.is_available():
            data_random = torch.ones(shape).cuda()
        else:
            data_random = torch.ones(shape)
        N = shape[0]
        sub_shape = shape[1:]
        for i in range(N):
            if random() >= self.p:
                continue
            if torch.cuda.is_available():
                data_random_ = (torch.rand(sub_shape).cuda() - 0.5) * 2 * self.amplitude
            else:
                data_random_ = (torch.rand(sub_shape) - 0.5) * 2 * self.amplitude
            data_random[i].add_(data_random_)
        return data_random

    def forward(self, input = None):
        if not self.training:
            return input
        shape = input.size()
        assert len(shape) >= 2 and len(shape) <= 5
        data_random = self.get_random_data(shape)
        output = PerturbationFunction.apply(input, data_random)
        return output



class Perturbation_(Module):
    """
    input: (N, C, L)
    output: L -- (N, C, L/2)
            H -- (N, C, L/2)
    """
    def __init__(self, p = 0.5, amplitude = 0.5):
        """
        :param p: 对每个数据样本进行扰动的概率，取值范围 【0,1】
        :param amplitude: 百分比，对每个数据样本进行扰动的幅度最大比例，建议取值范围： 【-1,1】
        """
        super(Perturbation_, self).__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.amplitude = amplitude

    def forward(self, input = None):
        if not self.training or random() >= self.p:
            return input
        shape = input.size()
        assert len(shape) >= 2 and len(shape) <= 5
        data_random = (torch.rand(shape) - 0.5) * 2 * self.amplitude + 1
        if torch.cuda.is_available():
            data_random = data_random.cuda()
        return input.mul(data_random)


class PerturbationFunction(Function):
    @staticmethod
    def forward(ctx, input, data_random):
        ctx.save_for_backward(data_random)
        output = input.mul(data_random)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        #print(type(grad_output))
        data_random, = ctx.saved_tensors
        grad_input = grad_output.mul(data_random)
        return grad_input, None

from torch.autograd import gradcheck

if __name__ == '__main__':


    """
    noise = torch.cuda.FloatTensor([4,1,2,3]) if torch.cuda.is_available() else torch.FloatTensor([4,1,2,3])
    feature = torch.rand([4,1,2,3], out = noise)
    #feature = torch.rand([4,1,2,3])
    print(feature.size())
    print(feature)
    feature = feature.transpose(dim0 = 0, dim1 = -1)
    print(feature.size())
    lines = torch.rand(4)
    print(lines)
    lines = lines.le(0.5).float()
    print(lines)
    feature_new = lines * feature #torch.matmul(lines, feature)
    feature_new = feature_new.transpose(dim0 = 0, dim1 = -1)
    print(feature_new.size())
    print(feature_new)
    
    input = np.array(range(24)).reshape((4,1,2,3,1))
    input_t = torch.Tensor(input)
    if torch.cuda.is_available():
        input_t = input_t.cuda()
    m = Perturbation()
    #m.eval()
    input_ = m.forward(input_t)
    #data_random = m.get_random_data(input_.size())
    data_random = (torch.rand(input_t.size()) - 0.5) * 2 * 0.5
    print(type(data_random))
    input_.requires_grad = True
    inp = (input_.double(), data_random.double())
    test = gradcheck(PerturbationFunction.apply, inp)
    print(test)
    """

    from torch.nn import Dropout2d
    input_t = torch.rand([256, 512, 7, 7])
    if torch.cuda.is_available():
        input_t = input_t.cuda()
    mm = Perturbation_o()
    m = Perturbation()
    n = Dropout2d()
    start = datetime.now()
    for i in range(10):
        o = mm(input_t)
    end = datetime.now()
    print('------------------------------------- mm took {} s'.format(end - start))
    start1 = datetime.now()
    for i in range(10):
        o = m(input_t)
    end1 = datetime.now()
    print('===================================== m took {} s'.format(end1 - start1))
    start1 = datetime.now()
    for i in range(10):
        o = n(input_t)
    end1 = datetime.now()
    print('===================================== n took {} s'.format(end1 - start1))

    #p = Perturbation()

    #print(p.get_random_data([4,1,2,3]))