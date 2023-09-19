import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.autograd.variable import *
import torch.nn as nn

LOSS_EST_MASK_KEY = 'est_mask'
LOSS_LABEL_MASK_KEY = 'label_mask'
LOSS_NFRAMES_KEY = 'nframes'

LOSS_EST_WAV_KEY = 'est_wav'
LOSS_LABEL_WAV_KEY = 'label_wav'
LOSS_MIX_WAV_KEY = 'mix_wav'

LOSS_EST_VAD_KEY = 'est_vad'
LOSS_LABEL_VAD_KEY = 'label_vad'


class LossInfo(object):
    def __init__(self, loss_info_dict):
        super(LossInfo, self).__init__()
        self.est_mask = loss_info_dict[LOSS_EST_MASK_KEY]
        self.label_mask = loss_info_dict[LOSS_LABEL_MASK_KEY]
        self.nframes = loss_info_dict[LOSS_NFRAMES_KEY]


class RawLossInfo(object):
    def __init__(self, raw_loss_info_dict):
        super(RawLossInfo, self).__init__()
        self.mix_wav = raw_loss_info_dict[LOSS_MIX_WAV_KEY]
        self.est_wav = raw_loss_info_dict[LOSS_EST_WAV_KEY]
        self.label_wav = raw_loss_info_dict[LOSS_LABEL_WAV_KEY]
        self.nframes = raw_loss_info_dict[LOSS_NFRAMES_KEY]


class VadLossInfo(object):
    def __init__(self, vad_info):
        super(VadLossInfo, self).__init__()
        self.est_vad = vad_info[LOSS_EST_VAD_KEY]
        self.label_vad = vad_info[LOSS_LABEL_VAD_KEY]
        self.nframes = vad_info[LOSS_NFRAMES_KEY]


class LossHelper(object):
    def __init__(self):
        super(LossHelper, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss()

    def dat_loss(self, dat_res, nframes):
        with torch.no_grad():
            w_for_loss_list = []
            for frame_num in nframes:
                w_for_loss_list.append(torch.ones(frame_num, dtype=torch.float32, device=TRAIN_DEVICE))
            w_for_loss = pad_sequence(w_for_loss_list * 2, batch_first=True)
            batch_size = dat_res.size(0)
            label = torch.cat([torch.ones_like(dat_res[:batch_size // 2, :]),
                               torch.zeros_like(dat_res[:batch_size // 2, :])], dim=0)
        loss = self.bce_loss(dat_res, label) * w_for_loss
        loss = loss.sum() / w_for_loss.sum()
        return loss

    def vad_loss(self, est_vad, label_vad, nframes):
        label_vad = label_vad.to(TRAIN_DEVICE).unsqueeze(dim=-1)
        label_frames = label_vad.size(1)
        with torch.no_grad():
            w_for_loss_list = []
            for frame_num in nframes:
                w_for_loss_list.append(torch.ones(frame_num, dtype=torch.float32, device=TRAIN_DEVICE))
            w_for_loss = pad_sequence(w_for_loss_list, batch_first=True).to(TRAIN_DEVICE)[:, :label_frames]
        est_vad = est_vad[:, :label_frames]
        loss = self.bce_loss(est_vad, label_vad) * w_for_loss.unsqueeze(dim=-1)
        loss = loss.sum() / w_for_loss.sum()
        with torch.no_grad():
            est = torch.sigmoid(est_vad)
            diff = (est - label_vad).abs()
            est_res = torch.where(diff < 0.5, torch.ones_like(diff), torch.zeros_like(diff)) * w_for_loss.unsqueeze(
                dim=-1)
            accuracy = est_res.sum() / w_for_loss.sum()
        return loss, accuracy

    def time_to_frq_loss(self, label, est, nframes):
        with torch.no_grad():
            mask_for_loss_list = []
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, LABEL_DIM, dtype=torch.float32, device=TRAIN_DEVICE))
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True)
            label_spec = self.stft.transform(label)
            mask_for_loss = mask_for_loss[:, :label_spec.size(1), :]
            label_r = label_spec[:, :, :, 0] * mask_for_loss
            label_i = label_spec[:, :, :, 1] * mask_for_loss
        est_spec = self.stft.transform(est)
        est_r = est_spec[:, :, :, 0] * mask_for_loss
        est_i = est_spec[:, :, :, 1] * mask_for_loss
        loss = torch.sum(((est_r.abs() + est_i.abs()) - (label_r.abs() + label_i.abs())).abs()) / torch.sum(
            mask_for_loss)
        return loss

    def frame_to_frq_loss(self, label_frame, est_frame, nframes):
        with torch.no_grad():
            mask_for_loss_list = []
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, LABEL_DIM, dtype=torch.float32, device=TRAIN_DEVICE))
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True)
            label_spec = torch.rfft(label_frame, signal_ndim=2)
            mask_for_loss = mask_for_loss[:, :label_spec.size(1), :]
            label_r = label_spec[:, :, :, 0] * mask_for_loss
            label_i = label_spec[:, :, :, 1] * mask_for_loss
        est_spec = torch.rfft(est_frame, signal_ndim=2)
        est_r = est_spec[:, :, :, 0] * mask_for_loss
        est_i = est_spec[:, :, :, 1] * mask_for_loss
        loss = torch.sum(((est_r.abs() + est_i.abs()) - (label_r.abs() + label_i.abs())).abs()) / torch.sum(
            mask_for_loss)
        return loss

    def gen_vad_loss(self, vad_info):
        est_vad = vad_info.est_vad.to(TRAIN_DEVICE)
        label_vad = vad_info.label_vad.to(TRAIN_DEVICE)
        nframes = vad_info.nframes
        useful_frames = torch.tensor(np.array(nframes), dtype=torch.float32, device=TRAIN_DEVICE)
        w_for_loss_list = []
        for frame_num in nframes:
            w_for_loss_list.append(torch.ones(frame_num, dtype=torch.float32, device=TRAIN_DEVICE))
        w_for_loss = pad_sequence(w_for_loss_list, batch_first=True)

        # vad = 0
        filter_loss = torch.mean(torch.sum(-(1 - label_vad) * torch.log(est_vad[:, :, 0] + EPSILON) * w_for_loss,
                                           dim=1) / useful_frames)
        # vad = 1
        target_loss = torch.mean(torch.sum(-label_vad * torch.log(est_vad[:, :, 1] + EPSILON) * w_for_loss,
                                           dim=1) / useful_frames)
        est_diff = torch.abs(label_vad - est_vad[:, :, 1])
        est_res = torch.where(est_diff < 0.5, torch.ones_like(est_diff), torch.zeros_like(est_diff)) * w_for_loss
        accuracy = torch.mean(torch.sum(est_res, dim=1) / useful_frames)
        loss = filter_loss + target_loss
        return loss, accuracy

    def gen_raw_loss(self, raw_loss_info):
        # return self._si_snr_loss(raw_loss_info.est_wav, raw_loss_info.label_wav, raw_loss_info.nframes)
        return self.sdr_raw_loss(raw_loss_info.mix_wav, raw_loss_info.est_wav, raw_loss_info.label_wav,
                                 raw_loss_info.nframes)

    def _raw_mse_loss(self, est_wav, label_wav, nframes):
        # from pytorch.tools.save_wav import save_info_to_mat
        # est_np = np.squeeze(est_wav.detach().cpu().numpy())
        # label_np = np.squeeze(label_wav.detach().cpu().numpy())
        # save_data = np.vstack((est_np, label_np))
        # save_info_to_mat('est_label_data', save_data)
        wav_len = []
        w_for_loss_list = []
        for frame_num in nframes:
            t_len = frame_num * WIN_OFFSET + WIN_LEN
            wav_len.append(t_len)
            w_for_loss_list.append(torch.ones(t_len, dtype=torch.float32))
        w_for_loss = pad_sequence(w_for_loss_list, batch_first=True)[:, :est_wav.size(-1)]
        assert (label_wav.size()[-1] - est_wav.size()[-1] < 320)
        label_wav = label_wav.cuda()[:, :est_wav.size()[-1]]
        w_for_loss = w_for_loss.cuda()[:, :est_wav.size()[-1]]
        mse = (est_wav - label_wav) * (est_wav - label_wav)
        mse = mse * w_for_loss
        mse_sum = torch.sum(mse, dim=1) / Variable(
            torch.FloatTensor(np.array(wav_len, dtype=np.float32))).cuda()
        loss = torch.mean(mse_sum)
        return loss

    def _raw_frame_mse_loss(self, est_wav, label_wav, nframes):
        # from pytorch.tools.save_wav import save_info_to_mat
        # est_np = np.squeeze(est_wav.detach().cpu().numpy())
        # label_np = np.squeeze(label_wav.detach().cpu().numpy())
        # save_data = np.vstack((est_np, label_np))
        # save_info_to_mat('est_label_data', save_data)
        # wav_len = []
        w_for_loss_list = []
        for frame_num in nframes:
            w_for_loss_list.append(torch.ones([frame_num, est_wav.size(2)], dtype=torch.float32))
        w_for_loss = pad_sequence(w_for_loss_list, batch_first=True)[:, :est_wav.size(1), :]
        w_for_loss = w_for_loss.cuda()[:, :est_wav.size(1), :]
        mask_label = label_wav * w_for_loss
        mask_est = est_wav * w_for_loss
        mse = (mask_label - mask_est) ** 2
        loss = torch.sum(mse) / torch.sum(w_for_loss)
        return loss

    def tas_mask_loss(self, mask_loss_power, frame_size, nframes):
        w_for_loss_list = []
        frame_list = []
        for frame_num in nframes:
            w_for_loss_list.append(
                torch.ones([frame_num * WIN_OFFSET // frame_size + 2, 161], dtype=torch.float32,
                           device=TRAIN_DEVICE))
            frame_list.append(frame_num * WIN_OFFSET // frame_size + 2)
        w_for_loss = pad_sequence(w_for_loss_list, batch_first=True)[:, :mask_loss_power.size(1), :]
        real_mask = w_for_loss * mask_loss_power
        mse_sum = torch.sum(real_mask.mean(dim=2), dim=1) / torch.from_numpy(
            np.array(frame_list, dtype=np.float32)).cuda()
        loss = torch.mean(mse_sum)
        return loss

    def sdr_raw_loss(self, mix, est, label, nframes):
        wav_len = []
        w_for_loss_list = []
        for frame_num in nframes:
            t_len = frame_num * WIN_LEN
            wav_len.append(t_len)
            w_for_loss_list.append(torch.ones(t_len, dtype=torch.float32))
        w_for_loss = pad_sequence(w_for_loss_list, batch_first=True)[:, :est.size(-1)]
        w_for_loss = w_for_loss.cuda()

        est = est * w_for_loss
        label = label.to(TRAIN_DEVICE)[:, :est.size(-1)]
        mix = mix.to(TRAIN_DEVICE)[:, :est.size(-1)]

        def _cal_sdr(real_data, est_data):
            return -torch.sum(real_data * est_data, dim=1) / (
                    (real_data ** 2).sum(dim=1).sqrt()
                    * (est_data ** 2).sum(dim=1).sqrt() + EPSILON)

        sdr_clean = _cal_sdr(label, est)
        sdr_noise = _cal_sdr(mix - label, mix - est)
        label_energy = (label ** 2).sum(dim=1)
        noise_energy = ((mix - label) ** 2).sum(dim=1)
        alpha = label_energy / (label_energy + noise_energy + EPSILON)

        loss = alpha * sdr_clean + (1 - alpha) * sdr_noise
        return loss.mean()

    def _si_snr_loss(self, est, label, nframes):
        batch = est.size(0)
        wav_len = []
        w_for_loss_list = []
        for frame_num in nframes:
            t_len = frame_num * WIN_LEN
            wav_len.append(t_len)
            w_for_loss_list.append(torch.ones(t_len, dtype=torch.float32))
        w_for_loss = pad_sequence(w_for_loss_list, batch_first=True)
        w_for_loss = w_for_loss.cuda()[:, :est.size(-1)]

        label = label.to(TRAIN_DEVICE)[:, :est.size()[-1]]
        est = torch.reshape(est, (batch, -1)) * w_for_loss
        label = torch.reshape(label, (batch, -1)) * w_for_loss

        zero_mean_est = est - torch.sum(est, 1, keepdim=True) / torch.sum(w_for_loss, 1, keepdim=True)
        zero_mean_label = label - torch.sum(label, 1, keepdim=True) / torch.sum(w_for_loss, 1, keepdim=True)

        pair_wise_dot = torch.sum(zero_mean_est * zero_mean_label, 1, keepdim=True)
        s_truth_energy = torch.sum(zero_mean_label ** 2, 1, keepdim=True) + EPSILON
        pair_wise_proj = pair_wise_dot * zero_mean_label / s_truth_energy
        e_noise = zero_mean_est - pair_wise_proj
        # shape is [B, nc, nc]
        pair_wise_snr = torch.div(torch.sum(pair_wise_proj ** 2, 1),
                                  torch.sum(e_noise ** 2, 1) + EPSILON)
        max_snr = 10 * torch.log10(pair_wise_snr + EPSILON)  # log operation use 10 as base

        return -torch.mean(max_snr)  # [timedomain, real(CRM), imag(CRM)]

    def gen_loss(self, loss_info):
        if isinstance(loss_info.est_mask, list):
            return self.combine_mse_loss(loss_info)
        else:
            return self.mse_loss(loss_info.est_mask, loss_info.label_mask, loss_info.nframes)

    def combine_mse_loss(self, loss_info):
        est_mask_8k, est_mask_16k = loss_info.est_mask
        loss_8k = self.mse_loss(est_mask_8k, loss_info.label_mask[:, :, :81], loss_info.mask_for_loss[:, :, :81],
                                loss_info.nframes)
        loss_16k = self.mse_loss(est_mask_16k, loss_info.label_mask, loss_info.mask_for_loss, loss_info.nframes)
        return loss_8k + loss_16k

    def mse_loss(self, est_mask, label_mask, nframes=None, label_dim=LABEL_DIM):
        if nframes is None:
            return ((est_mask - label_mask) ** 2).mean()
        else:
            with torch.no_grad():
                mask_for_loss_list = []
                for frame_num in nframes:
                    mask_for_loss_list.append(
                        torch.ones(frame_num, label_dim, dtype=torch.float32, device=TRAIN_DEVICE))
                mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True)
            loss = (((est_mask - label_mask) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
            return loss

    def mae_loss(self, est, label):
        return ((est - label) ** 2 + 1e-5).sqrt().mean()
        # return self.l1_loss(est, label)

    def spec_mse_loss(self, est_spec, label_spec, nframes):
        if len(est_spec.shape) != 4 or (est_spec.size(-1) != 2):
            raise Exception('shape error...')
        diff = est_spec - label_spec
        diff_r = diff[:, :, :, 0]
        diff_i = diff[:, :, :, 1]
        c_diff = diff_r ** 2 + diff_i ** 2

        with torch.no_grad():
            mask_for_loss_list = []
            for frame_num in nframes:
                mask_for_loss_list.append(torch.ones(frame_num, LABEL_DIM, dtype=torch.float32, device=TRAIN_DEVICE))
            mask_for_loss = pad_sequence(mask_for_loss_list, batch_first=True)

        masked_diff = c_diff * mask_for_loss
        loss = masked_diff.sum() / mask_for_loss.sum()
        return loss

    def mae_spec_loss(self, est_spec, speech_spec):
        mae_eps = 1e-7
        # |r - r'| + |i - i'| + |(r^2 + i^2).sqrt() - (r'^2 + r'^2).sqrt()|
        r_i_diff = ((est_spec - speech_spec) ** 2 + mae_eps).sqrt()
        r_i_loss = r_i_diff[..., 0].mean() + r_i_diff[..., 1].mean()
        mag_loss = ((est_spec[..., 0] ** 2 + est_spec[..., 1] ** 2 + mae_eps).sqrt() - (
                speech_spec[..., 0] ** 2 + speech_spec[..., 1] ** 2 + mae_eps).sqrt()).abs().mean()
        loss = r_i_loss + mag_loss

        # r_loss = self.l1_loss(est_spec[..., 0], speech_spec[..., 0])
        # i_loss = self.l1_loss(est_spec[..., 1], speech_spec[..., 1])
        # mag_loss = self.l1_loss((est_spec[..., 0] ** 2 + est_spec[..., 1] ** 2).sqrt(),
        #                         (speech_spec[..., 0] ** 2 + speech_spec[..., 1] ** 2).sqrt())
        # loss = r_loss + i_loss + mag_loss
        return loss

    def time_agc_loss(self, mix, est_c, nframes):
        with torch.no_grad():
            mix = mix.to(TRAIN_DEVICE)
            energy = (mix ** 2).sum(1)
            label_c_list = []
            frame_mask_list = []
            for i, frame_num in enumerate(nframes):
                frame_mask_list.append(torch.ones((frame_num, 1), device=TRAIN_DEVICE))
                c = (1 * frame_num * 160 / (energy[i] + EPSILON)).sqrt()
                label_c_list.append(c)
            frames_mask = pad_sequence(frame_mask_list, batch_first=True)
            label_c = torch.stack(label_c_list).reshape(-1, 1, 1)

        c_loss = (((est_c - label_c) ** 2 / label_c) * frames_mask).sum() / frames_mask.sum()
        loss = c_loss
        return loss
