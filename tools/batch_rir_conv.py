import torch
import torchaudio
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.signal as signal
from torch.fft import irfftn, rfftn

def complex_matmul(a, b, groups: int = 1):
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

def batch_rir_conv_same(data_ori, rir, n=1):
    '''
    data: [B, T]
    rir: [B, f_lens] or [B, C, f_lens]
    分段加速，避免fft太大
    '''
    b, t = data_ori.size()
    data_ori = data_ori.reshape(b * n, -1)
    if len(rir.shape) == 3:
        b, c, _ = rir.shape
        rir = torch.tile(rir.unsqueeze(1), [1, n, 1, 1]).reshape(b * n, c, -1)
    elif len(rir.shape) == 2:
        b, _ = rir.shape
        rir = torch.tile(rir.unsqueeze(1), [1, n, 1]).reshape(b * n, -1)
    res = batch_rir_conv(data_ori, rir)[..., :t // n]
    if len(rir.shape) == 3:
        res = res.reshape(b, n, c, -1).permute(0, 2, 1, 3).reshape(b, c, t)
    elif len(rir.shape) == 2:
        res = res.reshape(b, t)
    return res

def batch_rir_conv(data_ori, rir):
    '''
    data: [B, T]
    rir: [B, f_lens] or [B, C, f_lens]
    '''

    '''
    stupid implement
    # flip
    rir_w = torch.flip(rir, dims=[1]).unsqueeze(dim=1) # [B(out_channels), 1, f_lens]
    data = data_ori.unsqueeze(dim=0) # [1, B(in_channels), T]
    # pad data [1, B, 2*(k-1)+T]
    data = torch.cat([torch.zeros([1, data.size(1), rir_w.size(-1) - 1], dtype=data.dtype, device=data.device), data, 
                         torch.zeros([1, data.size(1), rir_w.size(-1) - 1], dtype=data.dtype, device=data.device)], dim=-1)
    out = F.conv1d(data, rir_w, groups=rir.size(0)).squeeze(dim=0)
    '''

    # k pad
    data_pad = F.pad(data_ori, [rir.size(-1) - 1, rir.size(-1) - 1])
    rir_pad = F.pad(rir, [0, data_pad.size(-1) - rir.size(-1)])
    data_fr = rfftn(data_pad, dim=-1)
    rir_fr = rfftn(rir_pad, dim=-1)
    if len(data_fr.shape) == 2 and len(rir_fr.shape) == 3:
        data_fr = data_fr.unsqueeze(dim=1)
    out_fr = data_fr * rir_fr
    out = irfftn(out_fr, dim=-1)
    out = out[..., rir.size(-1) - 1:]
    return out

def batch_lfilter(wav, b, a):
    '''
    wav: [B, T]
    b: [B, order+1]
    a: [B, order+1]
    '''
    out = torchaudio.functional.lfilter(wav, b, a)
    return out

def check_rir_conv():
    rdm_in = np.random.randn(16000 * 3)
    rdm_filter = np.random.randn(8000)
    out = signal.fftconvolve(rdm_in, rdm_filter, mode='full')

    rdm_in2 = np.random.randn(16000 * 3)
    rdm_filter2 = np.random.randn(8000)
    out2 = signal.fftconvolve(rdm_in2, rdm_filter2, mode='full')

    rdm_in3 = np.random.randn(16000 * 3)
    rdm_filter3 = np.random.randn(8000)
    out3 = signal.fftconvolve(rdm_in3, rdm_filter3, mode='full')

    torch_data = torch.from_numpy(np.stack([rdm_in, rdm_in2, rdm_in3], axis=0))
    torch_rir = torch.from_numpy(np.stack([rdm_filter, rdm_filter2, rdm_filter3], axis=0))

    out_torch = batch_rir_conv(torch_data, torch_rir)
    print(out)
    print(out_torch[0].detach().cpu().numpy())
    print(out2)
    print(out_torch[1].detach().cpu().numpy())
    print(out3)
    print(out_torch[2].detach().cpu().numpy())

if __name__ == '__main__':
    # import random
    # wav = np.random.randn(4, 16000)
    # wav = wav / np.max(np.abs(wav))
    # r1, r2, r3, r4 = (random.uniform(-3 / 8, 3 / 8) for i in range(4))
    # b = np.array([1.0, r1, r2])
    # a = np.array([1.0, r3, r4])
    # scipy_o = signal.lfilter(b, a, wav[0])

    # r1, r2, r3, r4 = (random.uniform(-3 / 8, 3 / 8) for i in range(4))
    # b2 = np.array([1.0, r1, r2])
    # a2 = np.array([1.0, r3, r4])
    # scipy_o2 = signal.lfilter(b2, a2, wav[1])

    # a = torch.from_numpy(np.stack([a, a2], axis=0))
    # b = torch.from_numpy(np.stack([b, b2], axis=0))

    # torch_o = batch_lfilter(torch.from_numpy(wav)[0:2], a, b)
    
    # print(scipy_o)
    # print(torch_o.detach().cpu().numpy()[0])
    # print(scipy_o2)
    # print(torch_o.detach().cpu().numpy()[1])

    check_rir_conv()