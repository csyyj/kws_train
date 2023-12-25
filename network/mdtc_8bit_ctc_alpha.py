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
        self.alpha = nn.Parameter(torch.FloatTensor(1, 257))
        nn.init.constant_(self.alpha, 3)
        # self.ln_0 = nn.LayerNorm([5, 4])
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
        PAD_LEN = 4
        xs_pad = F.pad(mag, [0, 0, PAD_LEN, 0])
        b, t, _ = mag.size()
        # xs_stack = torch.stack([xs_pad[:, i: i + t, 1:] for i in range(PAD_LEN + 1)], dim=2) # [B, T, 5, 64]
        # norm_xs = self.ln_0(xs_stack.reshape(b, t, PAD_LEN + 1, -1, 4).permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4).reshape(b, t, 5, -1)[:, :, -1]
        # norm_xs = F.pad(norm_xs, [1, 0])
        if self.training and False:
            avg_mag = torch.cumsum(mag, dim=1) / (torch.arange(t).reshape(1, t, 1) + 1).to(input_waveform.device)
        else:
            tmp = mag[:, 0]
            l = []
            for i in range(t):
                alpha = torch.sigmoid(self.alpha)
                tmp = alpha * tmp + (1 - alpha) * mag[:, i]
                l.append(tmp)
            avg_mag = torch.stack(l, dim=1)
                    
        norm_mag = mag / (avg_mag + 1e-8)
        abs_mel = torch.matmul(norm_mag, self.linear_to_mel_weight_matrix.to(input_waveform.device))
        abs_mel = abs_mel + 1e-6
        log_mel = abs_mel.log()
        # log_mel[log_mel < -6] = -6
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
        bias: bool = False,
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

    def forward(self, xs: torch.Tensor, xs_lens=None, cnn_cache=None, is_last_cache=False):
        if cnn_cache is None:
            # inputs = F.pad(xs, (self.receptive_fields, 0, 0, 0, 0, 0),
            #                 'constant')
            cnn_cache = torch.zeros([xs.shape[0], xs.shape[1], self.receptive_fields], dtype=xs.dtype, device=xs.device)
        inputs = torch.cat((cnn_cache, xs), dim=-1)
        if xs_lens is None or is_last_cache:
            new_cache = inputs[:, :, -self.receptive_fields:]
        else:
            new_cache = []
            for i, xs_len in enumerate(xs_lens):
                c = inputs[i:i+1, :, xs_len:xs_len+self.receptive_fields]
                new_cache.append(c)
            new_cache = torch.cat(new_cache, axis=0)
        # new_cache = inputs[:, :, -self.receptive_fields:]

        outputs1 = self.prelu1(self.conv1(inputs))
        outputs2 = self.conv2(outputs1)
        inputs = inputs[:, :, self.receptive_fields:]  
        if self.in_channels == self.res_channels:
            res_out = self.prelu2(outputs2 + inputs)
        else:
            res_out = self.prelu2(outputs2)
        return res_out, new_cache.detach(), [outputs1, outputs2, res_out]


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

    def forward(self, xs: torch.Tensor, xs_lens=None, cnn_caches=None, is_last_cache=False):
        new_caches = []
        out_list_for_loss = []
        for block, cnn_cache in zip(self.res_blocks, cnn_caches):
            xs, new_cache, out_l = block(xs, xs_lens, cnn_cache, is_last_cache=is_last_cache)
            new_caches.append(new_cache)
            out_list_for_loss += out_l
        return xs, new_caches, out_list_for_loss


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
        self.layer_norm = nn.LayerNorm([5, 2])
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
        self.pooling_cnn = nn.Conv2d(in_channels=stack_num, out_channels=1, kernel_size=(1, 1), bias=False)
        self.pooling_relu = nn.PReLU(1)
        self.stack_num = stack_num
        self.stack_size = stack_size
        
        self.fbank = Fbank(sample_rate=16000, filter_length=512, hop_length=256, n_mels=64)
        self.time_vad = TimeVad(256)
        vocab_size = 410 # 拼音分类
        self.pinyin_fc = torch.nn.Linear(res_channels, vocab_size)
        self.class_out = torch.nn.Linear(res_channels +  vocab_size, 16)
        self.ctc = CTC()
        self.drop_out = nn.Dropout(p=0.1)

    def forward(self, wav, kw_target=None, ckw_target=None, real_frames=None, label_frames=None, ckw_len=None, clean_speech=None, hidden=None, custom_in=None):
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
                
        
        xs = self.fbank(wav)
            
        b, t, f = xs.size()
        if random.random() < 0.5 and self.training:
            is_last_cache = True
        else:
            is_last_cache = False
        # PAD_LEN = 4
        # xs_pad = F.pad(xs, [0, 0, PAD_LEN, 0])
        # xs_stack = torch.stack([xs_pad[:, i: i + t] for i in range(PAD_LEN + 1)], dim=2) # [B, T, 5, 64]
        # norm_xs = self.layer_norm(xs_stack.reshape(b, t, PAD_LEN + 1, -1, 2).permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4).reshape(b, t, -1, 64)[:, :, -1]
        # outputs = norm_xs.transpose(1, 2)
        outputs = xs.transpose(1, 2)
        outputs_list = []
        outputs_list_for_loss = []
        outputs_cache_list  = []
        outputs, new_cache,  o_l_1= self.preprocessor(outputs, real_frames, hidden[0], is_last_cache=is_last_cache)
        outputs_list_for_loss += o_l_1
        outputs = self.prelu(outputs)
        outputs_list_for_loss.append(outputs)
        outputs_pre = outputs
        outputs_cache_list.append(new_cache)
        for i in range(len(self.blocks)):
            outputs, new_caches, o_l_tmp = self.blocks[i](outputs, real_frames, hidden[1+i*self.stack_size: 1 +(i+1)*self.stack_size], is_last_cache=is_last_cache)
            outputs_list_for_loss += o_l_tmp
            outputs_list.append(outputs)
            outputs_cache_list += new_caches
        # outputs = self.pooling_relu(self.pooling_cnn(torch.stack(outputs_list, dim=1))).squeeze(dim=1)
        outputs = sum(outputs_list)
        outputs_list_for_loss.append(outputs)
        outputs = outputs.transpose(1, 2)
        outputs = self.drop_out(outputs)
        pinyin_logist = self.pinyin_fc(outputs)
        # outputs_list_for_loss.append(pinyin_logist)
        logist = self.class_out(torch.cat([outputs, pinyin_logist], dim=-1))
        # outputs_list_for_loss.append(logist)

        if kw_target is not None:
            l2_loss = self.l2_regularization(l2_alpha=1)
            outputs_list.append(outputs_pre)
            l2_f_loss = self.l2_regularization_feature(outputs_list_for_loss)
            kws_loss, acc, vad_speech = self.max_pooling_loss_vad(logist, kw_target, clean_speech, ckw_len, real_frames, label_frames)
            ctc_loss = self.ctc(pinyin_logist, real_frames, ckw_target, ckw_len)
            loss = kws_loss + l2_f_loss + l2_loss + ctc_loss
            acc2 = 0
        else:
            loss = None
            acc = None
            acc2 = None
            vad_speech = None
        return logist, pinyin_logist, outputs_cache_list, loss, acc, acc2, vad_speech
    
    def max_pooling_loss_vad(self, logits_ori, target, clean_speech, ckw_len, real_frames, label_frames):
        label_vad, _ = self.time_vad(clean_speech)
        start_f = torch.argmax(label_vad, dim=1)
        last_f = label_vad.shape[1] - torch.argmax(torch.flip(label_vad, dims=[1]), dim=1) - 1
        logits_softmax = torch.softmax(logits_ori, dim=-1)
        num_utts = logits_ori.size(0)
        with torch.no_grad():
            max_logits, index = logits_softmax[:, :, 1:].max(1)
            num_correct = 0
            for i in range(num_utts):
                max_p, idx = max_logits[i].max(0)
                if max_p > 0.5 and (idx + 1 == target[i]):
                    num_correct += 1
                if max_p < 0.5 and target[i] == 0:
                    num_correct += 1
            acc = num_correct / num_utts
        logits = torch.clamp(torch.log_softmax(logits_ori, dim=-1), min=-8)
        from settings.config import TRAINING_KEY_WORDS
        acc_threshod = 1 / (len(TRAINING_KEY_WORDS) + 1) * 1.1
        clean_speech_vad = clean_speech.detach()

        loss = 0.0
        non_keyword_weight = 1.0
        keyword_weight = 1.0 #len(TRAINING_KEY_WORDS) + 1
        for i in range(num_utts):
            # 唤醒词
            if target[i] == 0:
                # 非唤醒词
                if acc > acc_threshod:
                    # 该惩罚项在早期启用会导致不收敛，输出全为background
                    prob = logits[i, :, 0]
                    min_prob = prob.sum()
                    # min_prob, _ = torch.min(prob, dim=0)
                    loss += (-min_prob) * non_keyword_weight
                # 查看醒词类别
                prob = logits[i, :, 1:]
                max_prob = torch.amax(prob)
                loss += max_prob * non_keyword_weight
            else:
                # 唤醒词
                prob = logits[i, :, target[i]]
                label_frame = label_frames[i]
                if label_frame > 1: # 非 oneshot
                    start = label_frame - 1
                    end = label_frame + 2
                else:
                    if start_f[i] < 10 or last_f[i] > real_frames[i] - 10:
                        start = 10
                        end = real_frames[i] - 10
                    else:
                        if ckw_len[i] <= 5: # 非oneeshot
                            start = (start_f[i] + last_f[i]) // 2 + 3
                            end = min(last_f[i] + 10, real_frames[i] - 10)
                            # if last_f[i] + 5 > real_frames[i] - 10: # vad 有问题了，降级
                            #     start = 10
                            #     end = real_frames[i] - 10
                            # else:
                            #     start = last_f[i]
                            #     end = min(last_f[i] + 20, real_frames[i] - 10)
                        else: # oneshot
                            start = start_f[i] + 30
                            end = min(last_f[i] - 20, real_frames[i] - 20)
                    if start >= end:
                        start = 10
                        end = real_frames[i] - 10
                    clean_speech_vad[i, :start] = 0
                    clean_speech_vad[i, end:] = 0
                prob1 = prob[start: end]
                max_prob, max_idx = torch.max(prob1, dim=0)
                loss += -max_prob * keyword_weight
                
                if acc > acc_threshod:
                    prob2 = prob[:start - 1]                    
                    prob2 = torch.amax(prob2, dim=0)
                    loss += prob2 * keyword_weight
                    
                    prob3 = prob[end + 1:]
                    prob3 = torch.amax(prob3, dim=0)
                    loss += prob3 * keyword_weight
                    # max_prob, max_idx = torch.max(prob1, dim=0)
                    # loss += -max_prob * keyword_weight
                    
                # other 
                if acc > acc_threshod:
                    # 该惩罚项在早期启用会导致不收敛，输出全为background
                    prob_other = torch.cat([logits[i, :, target[i] + 1:], logits[i, :, 1:target[i]]], dim=-1)                
                    max_prob_other = torch.amax(prob_other)
                    loss += max_prob_other * keyword_weight

        loss = loss / num_utts
        # Compute accuracy of current batch
        return loss, acc, clean_speech_vad
    
    def max_pooling_loss_vad_fast(self, logits, target, clean_speech, ckw_len, real_frames, label_frames):
        logits = torch.log_softmax(logits, dim=-1)
        b, t, c = logits.size()
        clean_speech_vad = clean_speech.detach()
        loss = 0.0
        non_k_scale = 4.0
        k_scale = 2.0
        logits = torch.clamp(logits, -8)
        kws_mask = torch.ones_like(target)
        kws_mask[target < 1] = 0
        # 非唤醒词
        non_kws_a = -(logits[:, :, 0] * (1 - kws_mask).reshape(b, 1)).sum() * non_k_scale / ((1 - kws_mask).sum() + 1e-5)  # 0 分类求和越大越好
        non_kws_b = (torch.amax(logits[:, :, 1:], dim=[1, 2]) * (1 - kws_mask).reshape(b)).sum() / ((1 - kws_mask).sum() + 1e-5) # 非 0 分类最大值越小越好
        # 唤醒词
        l = []
        ll = []
        for i in range(b):
            tmp = torch.cat([logits[i, :, max(target[i] + 1, 2):], logits[i, :, 1:target[i]]], dim=-1)
            l.append(tmp)
            ll.append(logits[i, :, target[i]])
        kws_logist = torch.stack(ll, dim=0)
        kws_a = (-torch.amax(kws_logist, dim=1) * kws_mask.reshape(b) * k_scale).sum() / ((kws_mask).sum() + 1e-5) # 目标分类最大值越大越好
        tmp = torch.stack(l, dim=0)
        kws_b = (torch.amax(tmp, dim=(1, 2)) * kws_mask.reshape(b) * k_scale).sum() / ((kws_mask).sum() + 1e-5) # 非目标分类最大值越小越好
        loss = (non_kws_a + non_kws_b +  kws_a + kws_b).mean()
        with torch.no_grad():
            logits_softmax = torch.softmax(logits, dim=-1)
            max_logits, index = logits_softmax[:, :, 1:].max(1)
            num_correct = 0
            for i in range(b):
                max_p, idx = max_logits[i].max(0)
                if max_p > 0.5 and (idx + 1 == target[i]):
                    num_correct += 1
                if max_p < 0.5 and target[i] == 0:
                    num_correct += 1
            acc = num_correct / b
        return loss, acc, clean_speech_vad
    
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
    y, _, _, _, _, _, _ = mdtc(x)  # batch-size * time * dim
    print('input shape: {}'.format(x.shape))
    print('output shape: {}'.format(y.shape))
    
