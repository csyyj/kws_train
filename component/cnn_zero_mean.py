import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, out_ac=None):
        super(Conv2DNorm, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_ac = out_ac
        self.weight = nn.Parameter(torch.FloatTensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        nn.init.normal_(self.weight, -3, 3)

    def forward(self, feat):
        log_feat = torch.log(torch.relu(feat) + 1e-7)
        zero_mean = self.weight - self.weight.mean(dim=(-3, -2, -1), keepdim=True)
        res = F.conv2d(log_feat, zero_mean, None, stride=self.stride, padding=self.padding)
        if self.out_ac is not None:
            res = self.out_ac(res)
        return res


class CnnZeroMean2d(torch.nn.Module):
    def __init__(self, input_channel=1, output_channel=8, kernel_size=(3, 2), stride=(1, 1), padding=None):
        super(CnnZeroMean2d, self).__init__()
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.cnn_weight = nn.Parameter(torch.FloatTensor(output_channel, input_channel, kernel_size[0], kernel_size[1]))
        nn.init.normal_(self.cnn_weight, 0, 1)
        self.cnn_bias = nn.Parameter(torch.FloatTensor(output_channel))
        nn.init.constant_(self.cnn_bias, 0)
        self.padding = padding

    def forward(self, feat):
        # expand channel dim
        if feat.size(1) != self.input_channel:
            if self.input_channel == 1:
                feat = feat.unsqueeze(1)
            else:
                assert 'feat dim not match'
        # feat = torch.unsqueeze(feat, 1)
        # convert to log
        with torch.no_grad():
            log_feat = torch.log(feat + 1e-4)
            log_feat[log_feat < -6] = -6

        zero_mean = self.cnn_weight - self.cnn_weight.mean(dim=(-3, -2, -1), keepdim=True)
        res = F.conv2d(log_feat, zero_mean, self.cnn_bias, stride=self.stride, padding=self.padding)
        return res


class CnnZeroMeanFrqDev(torch.nn.Module):
    def __init__(self, input_channel=1, output_channel=8, kernel_size=3):
        super(CnnZeroMeanFrqDev, self).__init__()
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.cnn_weight = nn.Parameter(torch.FloatTensor(output_channel, input_channel, 1, kernel_size))
        nn.init.normal_(self.cnn_weight, 0, 1)
        self.cnn_bias = nn.Parameter(torch.FloatTensor(output_channel))
        nn.init.constant_(self.cnn_bias, 0)

    def forward(self, feat):
        # expand channel dim
        if feat.size(1) != self.input_channel:
            if self.input_channel == 1:
                feat = feat.unsqueeze(1)
            else:
                assert 'feat dim not match'
        # feat = torch.unsqueeze(feat, 1)
        # convert to log
        with torch.no_grad():
            log_feat = torch.log(feat + 1e-7)
            # log_feat[log_feat < -6] = -6

        # expand left & right
        with torch.no_grad():
            log_feat = torch.cat([log_feat[..., -3:], log_feat, log_feat[..., 0:3]], dim=-1)
        zero_mean = self.cnn_weight - self.cnn_weight.mean(dim=-1, keepdim=True)
        res_1 = F.conv2d(log_feat, zero_mean, self.cnn_bias, stride=1)
        return res_1


class CnnZeroMeanTime(torch.nn.Module):
    def __init__(self, input_channel=1, output_channel=8, kernel_size=7, padding='same'):
        super(CnnZeroMeanTime, self).__init__()
        self.input_channel = input_channel
        self.cnn_weight = nn.Parameter(torch.FloatTensor(output_channel, input_channel, kernel_size, 1))
        nn.init.normal_(self.cnn_weight, 0, 1)
        self.cnn_bias = nn.Parameter(torch.FloatTensor(output_channel))
        nn.init.constant_(self.cnn_bias, 0)
        if padding == 'same':
            self.padding = kernel_size - 1
        else:
            self.padding = padding

    def forward(self, feat):
        # expand channel dim
        if feat.size(1) != self.input_channel:
            if self.input_channel == 1:
                feat = feat.unsqueeze(1)
            else:
                assert 'feat dim not match'
        # feat = torch.unsqueeze(feat, 1)
        # convert to log
        with torch.no_grad():
            log_feat = torch.log(feat + 1e-4)

        zero_mean = self.cnn_weight - self.cnn_weight.mean(dim=(-3, -2, -1), keepdim=True)
        res_1 = F.conv2d(log_feat, zero_mean, self.cnn_bias, stride=1, padding=(self.padding, 0))[:, :, :-self.padding]
        return res_1


if __name__ == '__main__':
    # net = CnnZeroMean2d(1, 2, kernel_size=(3, 3), stride=1, padding=0)
    net = Conv2DNorm(1, 2, kernel_size=(3, 3), stride=1, padding=0)
    in_data = torch.relu(torch.randn(1, 1, 3, 10))
    print(in_data)
    res = net(in_data)
    print(res.squeeze())
    res2 = net(in_data * 2)
    print(res2.squeeze())