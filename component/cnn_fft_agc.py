import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTAgc(torch.nn.Module):
    def __init__(self):
        super(FFTAgc, self).__init__()

    def forward(self, spec):
        with torch.no_grad():
            spec_pow = (spec ** 2).sum(-1)
            spec_pow = ((spec_pow[..., 1:-1] * 2).sum(-1) + spec_pow[..., 0] + spec_pow[..., -1]) / 320
            slice_gain = spec_pow.sqrt().unsqueeze(dim=-1).unsqueeze(dim=-1)
            for n in range(1, slice_gain.shape[1]):
                slice_gain[:, n, :] = slice_gain[:, n - 1, :] * 0.9 + slice_gain[:, n, :] * 0.1
            new_spec = spec / (slice_gain + 1e-3)
        return new_spec, slice_gain


class FFTMagAgc(torch.nn.Module):
    def __init__(self):
        super(FFTMagAgc, self).__init__()

    def forward(self, mag):
        with torch.no_grad():
            spec_pow = mag ** 2
            spec_pow = ((spec_pow[..., 1:-1] * 2).sum(-1) + spec_pow[..., 0] + spec_pow[..., -1]) / 320
            slice_gain = spec_pow.sqrt().unsqueeze(dim=-1)
            for n in range(1, slice_gain.shape[1]):
                slice_gain[:, n, :] = slice_gain[:, n - 1, :] * 0.9 + slice_gain[:, n, :] * 0.1
            new_mag = mag / (slice_gain + 1e-3)
        return new_mag, slice_gain
