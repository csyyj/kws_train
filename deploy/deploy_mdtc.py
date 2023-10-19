import os
import torch
import librosa
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def resume_model(net, models_path, device='cpu'):
    # if is_map_to_cpu or not torch.cuda.is_available():
    model_dict = torch.load(models_path, map_location=device)
    state_dict = model_dict['state_dict']
    for k, v in net.state_dict().items():
        if k.split('.')[0] == 'module':
            net_has_module = True
        else:
            net_has_module = False
        break
    dest_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split('.')[0] == 'module':
            ckpt_has_module = True
        else:
            ckpt_has_module = False
        if net_has_module == ckpt_has_module:
            dest_state_dict = state_dict
            break
        if ckpt_has_module:
            dest_state_dict[k.replace('module.', '')] = v
        else:
            dest_state_dict['module.{}'.format(k)] = v

    net.load_state_dict(dest_state_dict, False)
    step = model_dict['step']
    optim_state = model_dict['optimizer']
    print('finish to resume model {}.'.format(models_path))
    return step, optim_state

class STFT(torch.nn.Module):
    def __init__(self, win_len=1024, shift_len=512, window=None):
        super(STFT, self).__init__()
        if window is None:
            window = torch.from_numpy(np.sqrt(np.hanning(win_len).astype(np.float32)))
        self.win_len = win_len
        self.shift_len = shift_len
        self.window = window
    
    def transform(self, input_data):
        self.window = self.window.to(input_data.device)
        spec = torch.stft(input_data, n_fft=self.win_len, hop_length=self.shift_len, win_length=self.win_len, window=self.window, center=True, pad_mode='constant', return_complex=True)
        spec = torch.view_as_real(spec)
        return spec.permute(0, 2, 1, 3)
    
    def inverse(self, spec):
        self.window = self.window.to(spec.device)
        torch_wav = torch.istft(torch.view_as_complex(torch.permute(spec, [0, 2, 1, 3])), n_fft=self.win_len, hop_length=self.shift_len, win_length=self.win_len, window=self.window, center=True)
        return torch_wav
    
    def forward(self, input_data):
        stft_res = self.transform(input_data)
        reconstruction = self.inverse(stft_res)
        return reconstruction

class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        super().__init__()

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self, ys_hat: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    def softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


class Fbank(nn.Module):
    def __init__(self, sample_rate=16000, filter_length=512, hop_length=256, n_mels=64):
        super(Fbank, self).__init__()
        self.stft = STFT(filter_length, hop_length)
        self.linear_to_mel_weight_matrix = torch.from_numpy(librosa.filters.mel(sr=sample_rate,
                                                                                n_fft=filter_length,
                                                                                n_mels=n_mels,
                                                                                fmin=20,
                                                                                fmax=8000,
                                                                                htk=True,
                                                                                norm=None).T.astype(np.float32))

    def forward(self, input_waveform):
        spec = self.stft.transform(input_waveform) # [b, 201, 897]
        mag = (spec ** 2).sum(-1).sqrt()
        abs_mel = torch.matmul(mag, self.linear_to_mel_weight_matrix.to(input_waveform.device))
        abs_mel = abs_mel + 1e-4
        log_mel = abs_mel.log()
        log_mel[log_mel < -6] = -6
        return abs_mel



class DSDilatedConv1d(nn.Module):
    """Dilated Depthwise-Separable Convolution"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super(DSDilatedConv1d, self).__init__()
        self.receptive_fields = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(in_channels,
                                   out_channels // 2,
                                   kernel_size=1,
                                   padding=0,
                                   dilation=1,
                                   bias=bias)

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv(inputs)
        outputs = self.pointwise(outputs)
        return outputs


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        res_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool,
    ):
        super(TCNBlock, self).__init__()
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.receptive_fields = dilation * (kernel_size - 1)
        self.half_receptive_fields = self.receptive_fields // 2
        self.conv1 = DSDilatedConv1d(
            in_channels=in_channels,
            out_channels=res_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.prelu1 = nn.PReLU(res_channels // 2)

        self.conv2 = nn.Conv1d(in_channels=res_channels // 2,
                               out_channels=res_channels,
                               kernel_size=1)
        self.prelu2 = nn.PReLU(res_channels)

    def forward(self, xs: torch.Tensor, xs_lens=None, cnn_cache=None):
        if cnn_cache is None:
            # inputs = F.pad(xs, (self.receptive_fields, 0, 0, 0, 0, 0),
            #                 'constant')
            cnn_cache = torch.zeros([xs.shape[0], xs.shape[1], self.receptive_fields], dtype=xs.dtype, device=xs.device)
        inputs = torch.cat((cnn_cache, xs), dim=-1)
        if xs_lens is None:
            new_cache = inputs[:, :, -self.receptive_fields:]
        else:
            new_cache = []
            for i, xs_len in enumerate(xs_lens):
                c = inputs[i:i+1, :, xs_len:xs_len+self.receptive_fields]
                new_cache.append(c)
            new_cache = torch.cat(new_cache, axis=0)
        # new_cache = inputs[:, :, -self.receptive_fields:]

        outputs = self.prelu1(self.conv1(inputs))
        outputs = self.conv2(outputs)
        inputs = inputs[:, :, self.receptive_fields:]  
        if self.in_channels == self.res_channels:
            res_out = self.prelu2(outputs + inputs)
        else:
            res_out = self.prelu2(outputs)
        return res_out, new_cache.detach()


class TCNStack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stack_num: int,
        res_channels: int,
        kernel_size: int,
        causal: bool,
    ):
        super(TCNStack, self).__init__()
        assert causal is True
        self.in_channels = in_channels
        self.stack_num = stack_num
        # self.stack_size = stack_size
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.res_blocks = self.stack_tcn_blocks()
        self.receptive_fields = self.calculate_receptive_fields()
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def calculate_receptive_fields(self):
        receptive_fields = 0
        for block in self.res_blocks:
            receptive_fields += block.receptive_fields
        return receptive_fields

    def build_dilations(self):
        dilations = []
        # for s in range(0, self.stack_size):
        for l in range(0, self.stack_num):
            dilations.append(2**l)
        return dilations

    def stack_tcn_blocks(self):
        dilations = self.build_dilations()
        res_blocks = nn.ModuleList()

        res_blocks.append(
            TCNBlock(
                self.in_channels,
                self.res_channels,
                self.kernel_size,
                dilations[0],
                self.causal,
            ))
        for dilation in dilations[1:]:
            res_blocks.append(
                TCNBlock(
                    self.res_channels,
                    self.res_channels,
                    self.kernel_size,
                    dilation,
                    self.causal,
                ))
        return res_blocks

    def forward(self, xs: torch.Tensor, xs_lens=None, cnn_caches=None):
        new_caches = []
        for block, cnn_cache in zip(self.res_blocks, cnn_caches):
            xs, new_cache = block(xs, xs_lens, cnn_cache)
            new_caches.append(new_cache)
        return xs, new_caches


class MDTCSML(nn.Module):
    """Multi-scale Depthwise Temporal Convolution (MDTC).
    In MDTC, stacked depthwise one-dimensional (1-D) convolution with
    dilated connections is adopted to efficiently model long-range
    dependency of speech. With a large receptive field while
    keeping a small number of model parameters, the structure
    can model temporal context of speech effectively. It aslo
    extracts multi-scale features from different hidden layers
    of MDTC with different receptive fields.
    """
    def __init__(
        self,
        stack_num: int,
        stack_size: int,
        in_channels: int,
        res_channels: int,
        kernel_size: int,
        causal: bool,
    ):
        super(MDTCSML, self).__init__()
        self.layer_norm = nn.LayerNorm([1, 64])
        self.kernel_size = kernel_size
        self.causal = causal
        self.preprocessor = TCNBlock(in_channels,
                                     res_channels,
                                     kernel_size,
                                     dilation=1,
                                     causal=causal)
        self.prelu = nn.PReLU(res_channels)
        self.blocks = nn.ModuleList()
        self.receptive_fields = []
        self.receptive_fields.append(self.preprocessor.receptive_fields)
        for _ in range(stack_num):
            self.blocks.append(
                TCNStack(res_channels, stack_size, res_channels,
                         kernel_size, causal))
            self.receptive_fields.append(self.blocks[-1].receptive_fields)
        self.stack_num = stack_num
        self.stack_size = stack_size
        
        self.fbank = Fbank(sample_rate=16000, filter_length=512, hop_length=256, n_mels=64)
        vocab_size = 410 # 拼音分类
        self.pinyin_fc = torch.nn.Linear(res_channels, vocab_size)
        self.class_out_1 = torch.nn.Linear(res_channels +  vocab_size, 2)
        self.ctc = CTC()

        self.pinyin_embedding = nn.Parameter(torch.FloatTensor(vocab_size, 8), requires_grad=True)
        nn.init.normal_(self.pinyin_embedding, -1, 1)
        self.custom_class = nn.Linear(res_channels + vocab_size + 8 * 6, 2)

    def forward(self, wav, kw_target=None, ckw_target=None, real_frames=None, ckw_len=None, clean_speech=None, hidden=None, custom_in=None):
        if hidden is None:
            hidden = [None for _ in range(self.stack_size * self.stack_num + 1)]
        else:
            if hidden[0].shape[0] >= wav.shape[0]:
                b = wav.size(0)
                h_l = []
                for h in hidden:
                    h_l.append(h[:b])
                hidden = h_l
            else:
                hidden = [None for _ in range(self.stack_size * self.stack_num + 1)]
                
        with torch.no_grad():
            xs = self.fbank(wav)
        b, t, f = xs.size()
        # norm_xs = self.layer_norm(xs.reshape(b, t, 1, 64)).reshape(b, t, -1)
        outputs = xs.transpose(1, 2)
        outputs_list = []
        outputs_cache_list  = []
        outputs, new_cache = self.preprocessor(outputs, real_frames, hidden[0])
        outputs = self.prelu(outputs)
        outputs_pre = outputs
        outputs_cache_list.append(new_cache)
        for i in range(len(self.blocks)):
            outputs, new_caches = self.blocks[i](outputs, real_frames, hidden[1+i*self.stack_size: 1 +(i+1)*self.stack_size])
            outputs_list.append(outputs)
            outputs_cache_list += new_caches

        outputs = sum(outputs_list)
        outputs = outputs.transpose(1, 2)
        outputs = F.dropout(outputs, p=0.1)
        pinyin_logist = self.pinyin_fc(outputs)
        logist = self.class_out_1(torch.cat([outputs, pinyin_logist], dim=-1))
        logist = torch.softmax(logist, dim=-1)
        return logist

if __name__ == '__main__':
    THRES_HOLD = 0.5
    net_work = MDTCSML(stack_num=4, stack_size=4, in_channels=64, res_channels=128, kernel_size=7, causal=True)
    resume_model(net_work, './model/student_model/model-601500-1.0317201238870621.pickle')
    net_work.eval()
    import soundfile as sf
    data, _ = sf.read('./process/test_wav/nihaoaodi.wav')
    with torch.no_grad():
        data_in = torch.from_numpy(data.astype(np.float32)).reshape(1, -1)
        est_logist = net_work(data_in)
        est_logist = est_logist.squeeze()[:, 1]
    
        count = 0
        k = 0
        while k < est_logist.size(0):
            if est_logist[k] > THRES_HOLD:
                count += 1
                k += 30
            else:
                k += 1
        print(count)
    
