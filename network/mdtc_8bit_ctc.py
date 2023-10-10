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
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tools.torch_stft import STFT
from component.time_vad import TimeVad

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
        self.time_vad = TimeVad(256)
        vocab_size = 410 # 拼音分类
        self.pinyin_fc = torch.nn.Linear(res_channels, vocab_size)
        self.classifier = torch.nn.Linear(res_channels +  vocab_size, 3)
        self.ctc = CTC()

        self.pinyin_embedding = nn.Parameter(torch.FloatTensor(vocab_size, 8), requires_grad=True)
        nn.init.normal_(self.pinyin_embedding, -1, 1)
        self.custom_class = nn.Linear(res_channels + vocab_size + 8 * 6, 2)

    def forward(self, wav, kw_target=None, ckw_target=None, real_frames=None, ckw_len=None, clean_speech=None, hidden=None, custom_in=None):
        if hidden is None:
            hidden = [None for _ in range(self.stack_size * self.stack_num + 1)]
        else:
            if hidden[0].shape[0] > wav.shape[0]:
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
        logist = self.classifier(torch.cat([outputs, pinyin_logist], dim=-1))
        if custom_in is not None:
            embedding = torch.index_select(self.pinyin_embedding, dim=0, index=custom_in)
            embedding = torch.tile(embedding.reshape(b, 1, -1), [1, t, 1])
            feat = torch.cat([pinyin_logist, outputs, embedding], dim=-1)
            custom_logits = torch.softmax(self.custom_class(feat), dim=-1)

        if kw_target is not None:
            l2_loss = self.l2_regularization(l2_alpha=1)
            outputs_list.append(outputs_pre)
            l2_f_loss = self.l2_regularization_feature(outputs_list)
            kws_loss, acc = self.max_pooling_loss_vad(logist, kw_target, clean_speech, ckw_len)
            custom_loss, acc2 = self.custom_kws_loss(outputs, ckw_target, pinyin_logist)
            ctc_loss = self.ctc(pinyin_logist, real_frames, ckw_target, ckw_len)
            loss = kws_loss + l2_f_loss + l2_loss + 1 * ctc_loss + 0*custom_loss
        else:
            loss = None
            acc = None
            acc2 = None
        if custom_in is not None:
            return logist, pinyin_logist, outputs_cache_list, loss, acc, acc2
        else:
            return logist, pinyin_logist, outputs_cache_list, loss, acc, acc2

    def custom_kws_loss(self, encode_out, ckw_target, ys_hat):
        loss = 0.0
        non_keyword_weight = 4.0
        keyword_weight = 1.0
        embedding_l = []
        label_l = []
        b, t, f = encode_out.size()
        with torch.no_grad():
            ckw_target[ckw_target < 0] = 0
            for i in range(b):
                if random.random() < 0.5: # 选取 label 作为 embedding
                    idx = ckw_target[i].detach()[:6]
                    rdm_start = random.randint(4, 6)
                    idx[rdm_start:] = 0
                    embedding = torch.index_select(self.pinyin_embedding, dim=0, index=idx)
                    label = 1
                else: # 随机选取作为 embedding
                    perm = torch.randperm(self.pinyin_embedding.size(0)).to(encode_out.device)
                    idx = perm[:6]
                    if random.random() < 0.5:
                        idx[random.randint(4, 5):] = 0
                    embedding = torch.index_select(self.pinyin_embedding, dim=0, index=idx)
                    label = 0
                embedding_l.append(embedding)
                label_l.append(label)
            embedding = torch.tile(torch.stack(embedding_l, dim=0).reshape(b, 1, -1), [1, t, 1])
            perm = torch.randperm(b // 2).to(encode_out.device)
            embedding_new = embedding.clone()
            embedding_new[:b//2] = embedding[:b//2][perm]
        feat = torch.cat([ys_hat, encode_out, embedding_new], dim=-1)
        logits = torch.softmax(self.custom_class(feat), dim=-1)
        
        loss = 0
        for i in range(b):
            if label_l[i] == 1 and (embedding[i] - embedding_new[i]).abs().sum() < 1:
                # 唤醒词
                prob1 = logits[i, :, 1]
                prob1 = torch.clamp(prob1, 1e-8, 1.0)
                max_prob, max_idx = torch.max(prob1, dim=0)
                loss += -torch.log(max_prob) * keyword_weight
            else:
                # 非唤醒词
                prob = logits[i, :, 0]
                prob = torch.clamp(prob, 1e-8, 1.0)
                min_prob = prob.log().sum()
                loss += (-min_prob) * non_keyword_weight
                # 查看醒词类别
                prob = logits[i, :, 1:]
                prob = torch.clamp(prob, 1e-8, 1.0)
                max_prob = torch.amax(prob)
                loss += torch.log(max_prob) * non_keyword_weight
                

        loss = loss / b

        # Compute accuracy of current batch
        max_logits, index = logits[:, :, 1:].max(1)
        num_correct = 0
        for i in range(b):
            max_p, idx = max_logits[i].max(0)
            # Predict correct as the i'th keyword
            if max_p > 0.5 and (label_l[i] == 1):
                num_correct += 1
            # Predict correct as the filler, filler id < 0
            if max_p < 0.5 and (label_l[i] == 0):
                num_correct += 1
        acc = num_correct / b
        # acc = 0.0
        return loss, acc
    
    def max_pooling_loss_vad(self, logits, target, clean_speech, ckw_len):
        logits_softmax = torch.softmax(logits, dim=-1)
        logits = torch.log_softmax(logits, dim=-1)
        label_vad, _ = self.time_vad(clean_speech)
        start_f = torch.argmax(label_vad, dim=1)
        last_f = label_vad.shape[1] - torch.argmax(torch.flip(label_vad, dims=[1]), dim=1) - 1
        num_utts = logits.size(0)

        loss = 0.0
        non_keyword_weight = 2.0
        keyword_weight = 1.0
        for i in range(num_utts):
            # 唤醒词
            if target[i] == 0:
                # 非唤醒词
                prob = logits[i, :, 0]
                # prob = prob.masked_fill(mask[i], 1.0)
                prob = torch.clamp(prob, -8)
                min_prob = prob.sum()
                loss += (-min_prob) * non_keyword_weight
                # 查看醒词类别
                prob = logits[i, :, 1:]
                prob = torch.clamp(prob, -8)
                max_prob = torch.amax(prob)
                loss += max_prob * non_keyword_weight
            else:
                # 唤醒词
                prob = logits[i, :, target[i]]
                if random.random() < 0.01:
                    if ckw_len[i] < 5 and last_f[i] < logits.size(1) - 10: # 非 oneshot
                        start = last_f[i] - 2
                    else:
                        start = start_f[i] # oneshot 
                    # if start_f[i] <= 0 or last_f[i] >= logits.size(1):
                    #     print('start_f: {}, last_f: {}'.format(start_f[i], last_f[i]))
                    if ckw_len[i] > 5:
                        end = min(last_f[i] - 12, logits.size(1))
                    else:
                        end = min(last_f[i] + 3, logits.size(1))
                    prob1 = prob[start: end]
                    prob1 = torch.clamp(prob1, -8)
                    max_prob, max_idx = torch.max(prob1, dim=0)
                    loss += -max_prob * keyword_weight
                    
                    # prob2 = prob[:start]
                    # prob2 = 1 - prob2
                    # # prob2 = torch.clamp(prob2, 1e-8, 1.0)
                    # prob2 = prob2.sum()
                    # loss += -prob2
                    
                    # prob3 = prob[end:]
                    # prob3 = 1 - prob3
                    # # prob3 = torch.clamp(prob3, 1e-8, 1.0)
                    # prob3 = prob3.sum()
                    # loss += -prob3
                else:
                    prob1 = prob
                    prob1 = torch.clamp(prob1, -8)
                    max_prob, max_idx = torch.max(prob1, dim=0)
                    loss += -max_prob * keyword_weight


        loss = loss / num_utts

        # Compute accuracy of current batch
        max_logits, index = logits_softmax[:, :, 1:].max(1)
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
    
    def l2_regularization(self, l2_alpha=1):
        l2_loss = []
        for module in self.modules():
            if type(module) is nn.Conv2d or type(module) is nn.Conv1d or type(module) is nn.PReLU:
                l2_loss.append((module.weight ** 2).mean())
        return l2_alpha * torch.stack(l2_loss, dim=0).mean()
    
    def l2_regularization_feature(self, features, l2_alpha=0.01):
        l2_loss = []
        for f in features:
            l2_loss.append((f ** 2).mean())
        return l2_alpha * torch.stack(l2_loss, dim=0).mean()


if __name__ == '__main__':
    from thop import profile, clever_format
    mdtc = MDTCSML(stack_num=4, stack_size=4, in_channels=64, res_channels=128, kernel_size=7, causal=True)
    # print(mdtc)
    # torch.save({'state_dict': mdtc.state_dict()}, 'mdtc.pickle')
    num_params = sum(p.numel() for p in mdtc.parameters())
    print('the number of model params: {}'.format(num_params))
    x = torch.zeros(1, 160000)  # batch-size * time * dim
    total_ops, total_params = profile(mdtc, inputs=(x,), verbose=False)
    flops, params = clever_format([total_ops/10, total_params], "%.3f ")
    print(flops, params)

    target = torch.ones([1, 1], dtype=torch.long)
    clean_speech = torch.randn(1, 16000)
    y, _, _, _ = mdtc(x, target, clean_speech)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))
    
