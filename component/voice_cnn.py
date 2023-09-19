import torch
import numpy as np
from torch.nn import *
import torch.nn as nn


class VoiceBatchNorm2d(Module):
    def __init__(self, out_channels, out_width):
        super(VoiceBatchNorm2d, self).__init__()
        self.running_means = Parameter(torch.FloatTensor(1, out_channels, 1, out_width), requires_grad=False)
        nn.init.constant_(self.running_means, 0)
        self.running_vars = Parameter(torch.FloatTensor(1, out_channels, 1, out_width), requires_grad=False)
        nn.init.constant_(self.running_vars, 1)
        self.gamma = Parameter(torch.FloatTensor(1, out_channels, 1, out_width))
        nn.init.constant_(self.gamma, 1)
        self.beta = Parameter(torch.FloatTensor(1, out_channels, 1, out_width))
        nn.init.constant_(self.beta, 0)
        self.momentum = 0.1
        self.is_train = True

    def forward(self, input_feat, nframes):
        if self.training:  # only, when state is trainning, to update vars & means
            frames = torch.from_numpy(np.array(nframes, dtype=np.float32)).cuda()
            sum_frame = torch.sum(frames)
            input_mean = torch.sum(torch.sum(input_feat, dim=2, keepdim=True), dim=0, keepdim=True) / sum_frame
            input_var = torch.sum(torch.sum((input_feat - input_mean) ** 2, dim=2, keepdim=True), dim=0,
                                  keepdim=True) / sum_frame
            self.running_means.data = (1 - self.momentum) * self.running_means + self.momentum * input_mean
            self.running_vars.data = (1 - self.momentum) * self.running_vars + self.momentum * input_var
        eps = 1e-05
        res = (input_feat - self.running_means) / torch.sqrt(self.running_vars + eps) * self.gamma + self.beta
        return res


class VoiceConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_width, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(VoiceConv2d, self).__init__()
        self.cnn = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.gamma = Parameter(torch.FloatTensor(1, out_channels, 1, out_width))
        nn.init.constant_(self.gamma, 1)
        self.beta = Parameter(torch.FloatTensor(1, out_channels, 1, out_width))
        nn.init.constant_(self.beta, 0)

    def forward(self, input_feat):
        res = self.cnn(input_feat)
        res = res * self.gamma + self.beta
        return res


class VoiceConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, out_width, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1):
        super(VoiceConvTranspose2d, self).__init__()
        self.cnn_t = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, output_padding=output_padding, groups=groups,
                                        bias=bias, dilation=dilation)
        self.gamma = Parameter(torch.FloatTensor(1, out_channels, 1, out_width))
        nn.init.constant_(self.gamma, 1)
        self.beta = Parameter(torch.FloatTensor(1, out_channels, 1, out_width))
        nn.init.constant_(self.beta, 0)

    def forward(self, input_feat):
        res = self.cnn_t(input_feat)
        res = res * self.gamma + self.beta
        return res
