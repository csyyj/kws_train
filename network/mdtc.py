#!/usr/bin/env python3
# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import librosa
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tools.torch_stft import STFT
from component.time_vad import TimeVad

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
        return log_mel



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
        self.bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels,
                                   out_channels // 2,
                                   kernel_size=1,
                                   padding=0,
                                   dilation=1,
                                   bias=bias)

    def forward(self, inputs: torch.Tensor):
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
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
        self.bn1 = nn.BatchNorm1d(res_channels // 2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=res_channels // 2,
                               out_channels=res_channels,
                               kernel_size=1)
        self.bn2 = nn.BatchNorm1d(res_channels)
        self.relu2 = nn.ReLU()

    def forward(self, xs: torch.Tensor, xs_lens=None, cnn_cache=None):
        if cnn_cache is None:
            # inputs = F.pad(xs, (self.receptive_fields, 0, 0, 0, 0, 0),
            #                 'constant')
            cnn_cache = torch.zeros([xs.shape[0], xs.shape[1], self.receptive_fields], dtype=xs.dtype, device=xs.device)
        inputs = torch.cat((cnn_cache, xs), dim=-1)
        # if xs_lens is None:
        #     new_cache = inputs[:, :, -self.receptive_fields:]
        # else:
        #     new_cache = []
        #     for i, xs_len in enumerate(xs_lens):
        #         c = inputs[i:i+1, :, xs_len:xs_len+self.receptive_fields]
        #         new_cache.append(c)
        #     new_cache = torch.cat(new_cache, axis=0)
        new_cache = inputs[:, :, -self.receptive_fields:]

        outputs = self.relu1(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        inputs = inputs[:, :, self.receptive_fields:]  
        if self.in_channels == self.res_channels:
            res_out = self.relu2(outputs + inputs)
        else:
            res_out = self.relu2(outputs)
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
        self.kernel_size = kernel_size
        self.causal = causal
        self.preprocessor = TCNBlock(in_channels,
                                     res_channels,
                                     kernel_size,
                                     dilation=1,
                                     causal=causal)
        self.relu = nn.ReLU()
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
        self.classifier = torch.nn.Linear(res_channels, 2)
        
        self.fbank = Fbank(sample_rate=16000, filter_length=512, hop_length=256, n_mels=64)
        self.time_vad = TimeVad(256)

    def forward(self, wav, target=None, clean_speech=None, xs_lens=None, hidden=None):
        if hidden is None:
            hidden = [None for _ in range(self.stack_size * self.stack_num + 1)]
        with torch.no_grad():
            xs = self.fbank(wav)
        outputs = xs.transpose(1, 2)
        outputs_list = []
        outputs_cache_list  = []
        outputs, new_cache = self.preprocessor(outputs, xs_lens, hidden[0])
        outputs = self.relu(outputs)
        outputs_cache_list.append(new_cache)
        for i in range(len(self.blocks)):
            outputs, new_caches = self.blocks[i](outputs, xs_lens, hidden[1+i*self.stack_size: 1 +(i+1)*self.stack_size])
            outputs_list.append(outputs)
            outputs_cache_list += new_caches

        outputs = sum(outputs_list)
        outputs = outputs.transpose(1, 2)
        logist = self.classifier(outputs)
        if target is not None:
            loss, acc = self.max_pooling_loss_vad(logist, target, clean_speech)
        else:
            loss = None
            acc = None
        return logist, outputs_cache_list, loss, acc
    
    def max_pooling_loss_vad(self, logits, target, clean_speech):
        
        label_vad, _ = self.time_vad(clean_speech)
        start_f = torch.argmax(label_vad, dim=1)
        last_f = label_vad.shape[1] - torch.argmax(torch.flip(label_vad, dims=[1]), dim=1) - 1
        logits = torch.softmax(logits, dim=-1)
        num_utts = logits.size(0)

        loss = 0.0
        non_keyword_weight = 4.0
        keyword_weight = 1.0
        for i in range(num_utts):
            # 唤醒词
            if target[i] == 0:
                # 非唤醒词
                prob = logits[i, :, 0]
                # prob = prob.masked_fill(mask[i], 1.0)
                prob = torch.clamp(prob, 1e-8, 1.0)
                min_prob = prob.log().sum()
                loss += (-min_prob) * non_keyword_weight
            else:
                # 唤醒词
                prob = logits[i, :, target.squeeze(dim=1)[i]]
                prob1 = prob[start_f[i]: last_f[i] + 5]
                prob1 = torch.clamp(prob1, 1e-8, 1.0)
                max_prob, max_idx = torch.max(prob1, dim=0)
                loss += -torch.log(max_prob) * keyword_weight
                # prob_other = 1 - prob1
                # loss += -prob_other[:(max_idx - 3)].log().sum()
                # loss += -prob_other[(max_idx + 3):].log().sum()

                prob2 = prob[:start_f[i] - 2]
                prob2 = 1 - prob2
                prob2 = torch.clamp(prob2, 1e-8, 1.0)
                prob2 = prob2.log().sum()
                loss += -prob2

                prob3 = prob[last_f[i] + 5:]
                prob3 = 1 - prob3
                prob3 = torch.clamp(prob3, 1e-8, 1.0)
                prob3 = prob3.log().sum()
                loss += -prob3

        loss = loss / num_utts

        # Compute accuracy of current batch
        max_logits, index = logits[:, :, 1:].max(1)
        num_correct = 0
        for i in range(num_utts):
            max_p, idx = max_logits[i].max(0)
            # Predict correct as the i'th keyword
            if max_p > 0.5 and (idx + 1 == target[i]):
                num_correct += 1
            # Predict correct as the filler, filler id < 0
            if max_p < 0.5 and target[i] == 0:
                num_correct += 1
        acc = num_correct / num_utts
        # acc = 0.0
        return loss, acc


if __name__ == '__main__':
    from thop import profile, clever_format
    mdtc = MDTCSML(stack_num=4, stack_size=4, in_channels=64, res_channels=128, kernel_size=7, causal=True)
    # print(mdtc)
    # torch.save({'state_dict': mdtc.state_dict()}, 'mdtc.pickle')
    num_params = sum(p.numel() for p in mdtc.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(1, 16000)  # batch-size * time * dim
    total_ops, total_params = profile(mdtc, inputs=(x,), verbose=False)
    flops, params = clever_format([total_ops, total_params], "%.3f ")
    print(flops, params)

    target = torch.ones([1, 1], dtype=torch.long)
    clean_speech = torch.randn(1, 16000)
    y, _, _, _ = mdtc(x, target, clean_speech)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))
    
