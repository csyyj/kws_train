import torch
import torch.nn as nn


class FeatureNormMagOnline(nn.Module):
    def __init__(self, channels, feature_size):
        super(FeatureNormMagOnline, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(1, channels, 1, feature_size, 1), requires_grad=True)
        nn.init.constant_(self.weights, 1)
        self.bias = nn.Parameter(torch.FloatTensor(1, channels, 1, feature_size, 1), requires_grad=True)
        nn.init.constant_(self.bias, 0)
        self.alpha = nn.Parameter(torch.FloatTensor(1, channels, feature_size, 1), requires_grad=True)
        nn.init.constant_(self.alpha, -4)

    def forward(self, input, s_1=None):
        '''

        :param input: [B, C, T, F, 2]
        :return:
        '''
        # rms = torch.sqrt(torch.mean(input ** 2, dim=[1, 2], keepdim=True))
        # res = input / (rms + 1.0e-8) * self.weights + self.bias

        b, c, t, f, _ = input.size()
        in_data2 = torch.sum(input ** 2, dim=-1, keepdim=True)  # [B, C, T, F, 1]
        if s_1 is None:
            s_1 = torch.zeros([b, c, f, 1], dtype=input.dtype, device=input.device)
        l = []
        alpha = torch.sigmoid(self.alpha)
        for i in range(t):
            s_1 = s_1 * (1 - alpha) + in_data2[:, :, i] * alpha
            l.append(s_1)
        smooth_data = torch.stack(l, dim=2).sqrt()
        res = input / (smooth_data + 1e-8) * self.weights + self.bias
        return res, s_1


class FeatureNormMagOnlineOneMag(nn.Module):
    def __init__(self, channels, feature_size):
        super(FeatureNormMagOnlineOneMag, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(1, channels, 1, feature_size, 1), requires_grad=True)
        nn.init.constant_(self.weights, 1)
        self.bias = nn.Parameter(torch.FloatTensor(1, channels, 1, feature_size, 1), requires_grad=True)
        nn.init.constant_(self.bias, 0)
        self.alpha = nn.Parameter(torch.FloatTensor(1, 1, feature_size, 1), requires_grad=True)
        nn.init.constant_(self.alpha, -4)

    def forward(self, input, s_1=None):
        '''

        :param input: [B, C, T, F, 2]
        :return:
        '''
        # rms = torch.sqrt(torch.mean(input ** 2, dim=[1, 2], keepdim=True))
        # res = input / (rms + 1.0e-8) * self.weights + self.bias

        b, c, t, f, _ = input.size()
        in_data2 = torch.sum(input ** 2, dim=-1, keepdim=True)  # [B, C, T, F, 1]
        if s_1 is None:
            s_1 = torch.zeros([b, 1, f, 1], dtype=input.dtype, device=input.device)
        l = []
        alpha = torch.sigmoid(self.alpha)
        for i in range(t):
            s_1 = s_1 * (1 - alpha) + in_data2[:, 0:1, i] * alpha
            l.append(s_1)
        smooth_data = torch.stack(l, dim=2).sqrt()
        cat_data = torch.cat([smooth_data, in_data2[:, 1:].sqrt()], dim=1)
        res = input / (cat_data + 1e-8) * self.weights + self.bias
        return res, s_1, smooth_data


class FeatureNormSingle(nn.Module):
    def __init__(self, feature_size):
        super(FeatureNormSingle, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(1, 1, feature_size), requires_grad=True)
        nn.init.constant_(self.weights, 1)
        self.bias = nn.Parameter(torch.FloatTensor(1, 1, feature_size), requires_grad=True)
        nn.init.constant_(self.bias, 0)
        self.alpha = nn.Parameter(torch.FloatTensor(1, feature_size), requires_grad=True)
        nn.init.constant_(self.alpha, -4)

    def forward(self, mag, s=None):
        '''

        :param input: [B, T, F]
        :return:
        '''

        b, t, f = mag.size()
        power = mag ** 2
        if s is None:
            s = torch.zeros([b, f], dtype=mag.dtype, device=mag.device)
        l = []
        alpha = torch.sigmoid(self.alpha)
        for i in range(t):
            s = s * (1 - alpha) + power[:, i] * alpha
            l.append(s)
        smooth_data = torch.stack(l, dim=1).sqrt()  # [b, t, f]
        res = mag / (smooth_data + 1e-8) * self.weights + self.bias
        return res


if __name__ == '__main__':
    import numpy as np

    norm = FeatureNormSingle(257)
    in_data = torch.abs(torch.randn(1, 100, 257))
    res = norm(in_data)
    res2 = norm(in_data * 10)
    print('diff: {}'.format(np.max(np.abs((res - res2).detach().cpu().numpy()))))
