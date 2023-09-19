import os
import random
from typing import Any
import torch.nn as nn
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import *
import scipy.io as sio
import scipy.signal as signal
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from scipy.signal import stft, istft
from tools.stft_istft import STFT
from tools.batch_rir_conv import batch_rir_conv_same


EPSILON = 1e-7
NOISE_PARTS_NUM = 20

#
S0_KEY = 's0'
S1_KEY = 's1'
S2_KEY = 's2'
S0_RIR = 's0_rir'
S1_RIR = 's1_rir'
S2_RIR = 's2_rir'
ANCHOR_KEY = 'anchor'
NOISE_KEY = 'noise'
MIX_KEY = 'mix'
NOISE_RIR = 'n_rir'
DIFFUSE_NOISE = 'diffuse_noise'
SPKID = 'spkid'



def gen_target_file_list(target_dir, target_ext='.wav'):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == target_ext and '._' not in f:
                l.append(f)
    return l



class SECSMDataset(Dataset):
    def __init__(self, clean_s_dir, noisy_s_dir, noise_dir, rir_path, diffuse_noise_dir):
        super(Dataset, self).__init__()
        self.wav_len = 16000 * 50
        self.anchor_len = 16000 * 10
        self.clean_speaker_list = self.gen_speaker_list(clean_s_dir)
        self.noisy_speaker_list = self.gen_speaker_list(noisy_s_dir)
        self.noise_data_info = self._list_noise_and_snr(noise_dir)
        self.rir_list = gen_target_file_list(rir_path, '.npy')
        self.diffuse_noise_list = gen_target_file_list(diffuse_noise_dir, '.npy')
    
    def gen_speaker_list(self, speech_dir):
        wav_l = []
        for root, dirs, files in os.walk(speech_dir, followlinks=True):
            for f in files:
                f = os.path.join(root, f)
                ext = os.path.splitext(f)[1]
                ext = ext.lower()
                if ext == '.npy' and '._' not in f and ('elevoc_vad' not in f):
                    wav_l.append(f)
        return wav_l

    def __getitem__(self, idx):
        # point noise
        n = self._read_noise(noise_len=self.wav_len)
        # target
        s_0, anchor, spk_id_0, is_clean = self._get_long_wav(is_need_anchor=True)
        while max(abs(s_0)) < 0.01:
            s_0, anchor, spk_id_0, is_clean = self._get_long_wav(is_need_anchor=True)
        
        if random.random() < 1:
            s_1, _, spk_id_1, is_clean = self._get_long_wav(is_need_anchor=True)
            while max(abs(s_1)) < 0.01 or spk_id_1 == spk_id_0:
                s_1, _, spk_id_1, _ = self._get_long_wav(is_need_anchor=True)
        else:
            spk_id_1 = -1
            s_1 = np.zeros_like(s_0)
        
        if random.random() < 0:
            s_2, _, spk_id_2, is_clean = self._get_long_wav(is_need_anchor=True)
            while max(abs(s_2)) < 0.01 or spk_id_2 == spk_id_0 or spk_id_2 == spk_id_1:
                s_2, _, spk_id_2, _ = self._get_long_wav(is_need_anchor=True)
        else:
            s_2 = np.zeros_like(s_0)
        

        while True:
            try:
                rir_path = self.rir_list[random.randint(0, len(self.rir_list) - 1)]                
                pid = os.getpid()
                # print('PID:{}---{}'.format(pid, rir_path))
                p_rir, s0_rir, s1_rir, s2_rir = self._load_rir(rir_path)
                if np.sum(np.abs(s0_rir)) < 1e-4:
                    print(rir_path)
                    self.rir_list.remove(rir_path)
                    os.remove(rir_path)
                    continue
            except:
                print(rir_path)
            else:
                break
        
        d_n_path = self.diffuse_noise_list[random.randint(0, len(self.diffuse_noise_list) - 1)]
        d_n = np.load(d_n_path, mmap_mode='c')
        rdm_start = random.randint(0, d_n.shape[0] - 1 - s_0.size)
        d_n = d_n[rdm_start:rdm_start + s_0.size, random.randint(0, d_n.shape[1] - 1)]

        res_dict = {SPKID: np.array([spk_id_0]).astype(np.int64),
                    S0_KEY: s_0.astype(np.float32),
                    S0_RIR: s0_rir.astype(np.float32),
                    S1_KEY: s_1.astype(np.float32),
                    S1_RIR: s1_rir.astype(np.float32),
                    S2_KEY: s_2.astype(np.float32),
                    S2_RIR: s2_rir.astype(np.float32),
                    ANCHOR_KEY: anchor.astype(np.float32),
                    NOISE_KEY: n.astype(np.float32),
                    NOISE_RIR: p_rir.astype(np.float32),
                    DIFFUSE_NOISE: d_n.astype(np.float32)
                    }
        return res_dict

    def _read_noise(self, noise_len):
        noise_data_max_index = len(self.noise_data_info) - 1
        noise_data = self.noise_data_info[random.randint(0, noise_data_max_index)]

        data_parts = noise_data.shape[0]
        data_len = noise_data.shape[1]

        n_0, n_1 = random.randint(0, data_parts - 1), random.randint(0, data_len - noise_len - 1)
        n = noise_data[n_0:n_0 + 1, n_1:n_1 + noise_len]
        while np.sum(np.abs(n)) < 1:
            n_0, n_1 = random.randint(0, data_parts - 1), random.randint(0, data_len - noise_len - 1)
            n = noise_data[n_0:n_0 + 1, n_1:n_1 + noise_len]
        n = np.squeeze(n).astype(np.float32)
        return n

    def _get_long_wav(self, is_need_anchor=False):
        while True:
            spk_id = random.randint(0, len(self.clean_speaker_list) -1)
            if spk_id <= len(self.clean_speaker_list) - 1:
                is_clean = np.array([1.0]).astype(np.float32)
                spk_path = self.clean_speaker_list[spk_id]
            else:
                spk_path = self.noisy_speaker_list[spk_id - len(self.clean_speaker_list)]
                is_clean = np.array([0.0]).astype(np.float32)
            spk_wav = np.load(spk_path, mmap_mode='c')
            if spk_wav.size - 1 - self.wav_len > 0:
                break
        rdm_start = random.randint(0, spk_wav.size - 1 - self.wav_len)
        sel_wav = spk_wav[rdm_start:rdm_start + self.wav_len]
        if is_need_anchor:
            rdm_start = random.randint(0, spk_wav.size - 1 - self.anchor_len)
            anchor_wav = spk_wav[rdm_start:rdm_start + self.anchor_len]
            return sel_wav.astype(np.float32), anchor_wav, spk_id, is_clean
        else:
            return sel_wav.astype(np.float32), spk_id, is_clean

    def _list_noise_and_snr(self, noise_path):
        noise_bin_ext = '.bin'
        noise_info = []
        for f in os.listdir(noise_path):
            full_path = os.path.join(noise_path, f)
            if os.path.isfile(full_path) and os.path.splitext(f)[1] == noise_bin_ext:
                noise_np = np.memmap(full_path, dtype='float32', mode='c')
                noise_np = noise_np[:noise_np.shape[0] // NOISE_PARTS_NUM * NOISE_PARTS_NUM]
                noise_data = np.reshape(noise_np, (NOISE_PARTS_NUM, noise_np.shape[0] // NOISE_PARTS_NUM))
                noise_info.append((noise_data))
        return noise_info
    
    def __len__(self):
        return len(self.noisy_speaker_list) * 10000

    def _load_rir(self, path):
        rir = np.load(path, mmap_mode='c')
        rir_pad = np.zeros([rir.shape[0], rir.shape[1], 4000], dtype=np.float32)
        rir_pad[..., :min(4000, rir.shape[-1])] = rir[..., :4000]
        rir_pad = np.squeeze(rir_pad,axis=1)
        p_rir = rir_pad[0]
        s0_rir = rir_pad[1]
        s1_rir = rir_pad[2]
        s2_rir = rir_pad[3]
        return p_rir, s0_rir, s1_rir, s2_rir


class BatchDataLoader(DataLoader):
    __metaclass__ = ABCMeta

    def __init__(self, s_mix_dataset, batch_size, is_shuffle=False, workers_num=0, sampler=None):
        super(BatchDataLoader, self).__init__(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     num_workers=workers_num,
                                     collate_fn=self.collate_fn, sampler=sampler)

    @staticmethod
    @abstractmethod
    def gen_batch_data(zip_res):
        return zip_res

    @staticmethod
    def collate_fn(batch):
        # batch.sort(key=lambda x: x[SPEECH_RIR].shape[0], reverse=True)
        keys = batch[0].keys()
        parse_res = []
        for item in batch:
            l = []
            for key in keys:
                l.append(item[key])
            parse_res.append(l)
        zip_res = list(zip(*parse_res))
        batch_dict = {}
        for i, key in enumerate(keys):
            data = torch.from_numpy(np.array(list(zip_res[i])))
            if isinstance(data[0], torch.Tensor):
                data = pad_sequence(data, batch_first=True)
            batch_dict[key] = data
        batch_info = SMBatchInfo(batch_dict)
        return batch_info

class GPUDataSimulate(nn.Module):
    def __init__(self, train_frq_response, inter_snr_list, point_snr_list, diffuse_snr_list, device='cpu'):
        super(GPUDataSimulate, self).__init__()
        self.inter_snr_list = inter_snr_list
        self.point_snr_list = point_snr_list
        self.diffuse_snr_list = diffuse_snr_list
        self.device = device
        self.hpf = self.gen_hpf()
        frq_response = np.load(train_frq_response, mmap_mode='c')
        self.frq_response = torch.from_numpy(frq_response.T)
        self.stft = STFT(512, 256)
        self.b = torch.from_numpy(np.array([1.0, -2, 1], dtype=np.float32)).unsqueeze(dim=0)
        self.a = torch.from_numpy(np.array([1.0, -1.99599, 0.99600], dtype=np.float32)).unsqueeze(dim=0)

    def __call__(self, batch_info) -> Any:
        # with torch.cuda.stream(self.stream):
        spkid, anchor, s0, s0_rir, s1, s1_rir, s2, s2_rir, n, n_rir, dn = \
            batch_info.spkid, batch_info.anchor, batch_info.s0, batch_info.s0_rir, batch_info.s1, batch_info.s1_rir, batch_info.s2, batch_info.s2_rir, \
            batch_info.n, batch_info.n_rir, batch_info.diffuse_noise
        anchor_mix, anchor_tgt, s0_tgt, s1_tgt, s2_tgt, mix, spkid, spk_tgt = self.simulate_data(spkid, anchor, s0, s0_rir, s1, s1_rir, s2, s2_rir, n, n_rir, dn)
        return anchor_mix, mix, s0_tgt, s1_tgt, s2_tgt, spkid, spk_tgt


    def gen_hpf(self):
        s = np.zeros(256)
        s[0] = 1.0
        b = np.array([1.0, -2, 1])
        a = np.array([1.0, -1.99599, 0.99600])
        hpf = signal.lfilter(b, a, s).astype(np.float32)
        hpf = torch.from_numpy(hpf)
        return hpf
    
    def simulate_freq_response(self, wav, firs=None, is_need_fir=False):
        # hpf
        if self.device != self.hpf.device:
            self.hpf = self.hpf.to(self.device)
            self.frq_response = self.frq_response.to(self.device)
        wav = batch_rir_conv_same(wav, torch.tile(self.hpf.unsqueeze(dim=0), [wav.size(0), 1]))
        if firs is None:
            idxs = np.random.choice(self.frq_response.shape[0], wav.size(0), replace=False)
            firs = self.frq_response[idxs]
        wav = batch_rir_conv_same(wav, firs)
        if is_need_fir:
            return wav, firs
        else:
            return wav
    
    def set_zeros(self, wavs, wavs2=None):
        for i in range(wavs.size(0)):
            rdm_times = random.randint(3, 10)
            rdm_index_l = []
            for k in range(rdm_times):
                if k == 0:
                    if random.random() < 0.1:
                        rdm_index_l.append(0)
                    else:
                        rdm_index_l.append(random.randint(0,  wavs.size(-1) - 1600))
                else:
                    rdm_index_l.append(random.randint(0,  wavs.size(-1) - 1600))
            start_index_l = sorted(rdm_index_l)
            start_index_l.append(wavs.size(-1))
            for j in range(len(start_index_l) - 1):
                start = start_index_l[j]
                end = random.randint(start, start_index_l[j + 1])
                wavs[i, start:end] = 0
                if wavs2 is not None:
                    wavs2[i, start:end] = 0
        return wavs, wavs2

    # @torch.compile
    def simulate_data(self, spkid, anchor, s0, s0_rir, s1, s1_rir, s2, s2_rir, n, n_rir, dn):
        with torch.no_grad():
            spkid = spkid.to(self.device)
            anchor = anchor.to(self.device)
            s0 = s0.to(self.device)
            s0_rir = s0_rir.to(self.device)
            s1 = s1.to(self.device)
            s1_rir = s1_rir.to(self.device)
            s2 = s2.to(self.device)
            s2_rir = s2_rir.to(self.device)
            n = n.to(self.device)
            n_rir = n_rir.to(self.device)
            dn = dn.to(self.device)
        
            
            anchor_rsp, firs = self.simulate_freq_response(anchor, is_need_fir=True)
            anchor_rir = s0_rir[torch.randperm(s0_rir.size(0))]
            anchor = batch_rir_conv_same(anchor_rsp, anchor_rir)
            anchor_rir[..., 640:] = 0
            anchor_tgt = batch_rir_conv_same(anchor_rsp, anchor_rir)
            
            s0 = self.simulate_freq_response(s0, firs=firs)
            s0_rev = batch_rir_conv_same(s0, s0_rir)
            s0_rir[..., 640:] = 0
            s0_tgt = batch_rir_conv_same(s0, s0_rir)
            
            s1 = self.simulate_freq_response(s1)
            s1_rev = batch_rir_conv_same(s1, s1_rir)
            s1_rir[..., 640:] = 0
            s1_tgt = batch_rir_conv_same(s1, s1_rir)
            
            s2 = self.simulate_freq_response(s2)
            s2_rev = batch_rir_conv_same(s2, s2_rir)
            s2_rir[..., 640:] = 0
            s2_tgt = batch_rir_conv_same(s2, s2_rir)


            n = self.simulate_freq_response(n)
            n_rev = batch_rir_conv_same(n, n_rir)

            s0_power = (s0_tgt ** 2).sum(-1)
            s1_power = (s1_tgt ** 2).sum(-1)
            s2_power = (s2_tgt ** 2).sum(-1)
            n_power = (n_rev ** 2).sum(-1)
            diffuse_power = (dn ** 2).sum(-1)

            n_snr = torch.randint(self.point_snr_list[0], self.point_snr_list[-1], (s0_power.size(0),), dtype=s0.dtype,
                                        device=s0.device)
            n_alpha = (s0_power / (EPSILON + n_power * (10.0 ** (n_snr / 10.0)))).sqrt().unsqueeze(dim=-1)
            
            s1_snr = torch.randint(self.inter_snr_list[0], self.inter_snr_list[-1], (s0_power.size(0),), dtype=s0.dtype,
                                        device=s0.device)
            s1_alpha = (s0_power / (EPSILON + s1_power * (10.0 ** (s1_snr / 10.0)))).sqrt().unsqueeze(dim=-1)

            s2_snr = torch.randint(self.inter_snr_list[0], self.inter_snr_list[-1], (s0_power.size(0),), dtype=s0.dtype,
                                        device=s0.device)
            s2_alpha = (s0_power / (EPSILON + s2_power * (10.0 ** (s2_snr / 10.0)))).sqrt().unsqueeze(dim=-1)
            
            diffuse_snr = torch.randint(self.inter_snr_list[0], self.inter_snr_list[-1], (s0_power.size(0),), dtype=s0.dtype,
                                        device=s0.device)
            diffuse_alpha = (s0_power / (EPSILON + diffuse_power * (10.0 ** (diffuse_snr / 10.0)))).sqrt().unsqueeze(dim=-1)

            s0_rev, s0_tgt = self.set_zeros(s0_rev, s0_tgt)
            s1_rev, s1_tgt = self.set_zeros(s1_rev, s1_tgt)
            s2_rev, s2_tgt = self.set_zeros(s2_rev, s2_tgt)

            mix = s0_rev + s1_rev * s1_alpha + s2_rev * s2_alpha + n_alpha * n_rev + diffuse_alpha * dn[:, :n_rev.size(1)]
            mix_amp = torch.rand(s0_rev.size(0), 1, device=s0_rev.device)
            alpha = 1 / (torch.amax(torch.abs(mix), -1, keepdim=True) + 1e-6) * mix_amp
            mix *= alpha
            s0_tgt *= alpha
            s1_tgt *= alpha * s1_alpha
            s2_tgt *= alpha * s2_alpha

            anchor_power = (anchor ** 2).sum(-1)
            dn = dn[torch.randperm(s0_rir.size(0))]
            diffuse_power = (dn ** 2).sum(-1)
            an_diffuse_snr = torch.randint(self.diffuse_snr_list[0], self.diffuse_snr_list[-1], (s0_power.size(0),), dtype=s0.dtype,
                                        device=s0.device)
            an_diffuse_alpha = (anchor_power / (EPSILON + diffuse_power * (10.0 ** (an_diffuse_snr / 10.0)))).sqrt().unsqueeze(dim=-1)
            anchor_mix = anchor + an_diffuse_alpha * dn[:, :anchor.size(1)]
            # ns_tgt = s0_tgt + s1_tgt + s2_tgt
            return anchor_mix, anchor_tgt, s0_tgt, s1_tgt, s2_tgt, mix, spkid, s0_tgt


class SMBatchInfo(object):
    def __init__(self, batch_dict):
        super(SMBatchInfo, self).__init__()
        self.spkid = batch_dict[SPKID] if SPKID in batch_dict else None
        self.s0 = batch_dict[S0_KEY] if S0_KEY in batch_dict else None
        self.s0_rir = batch_dict[S0_RIR] if S0_RIR in batch_dict else None
        self.s1 = batch_dict[S1_KEY] if S1_KEY in batch_dict else None
        self.s1_rir = batch_dict[S1_RIR] if S1_RIR in batch_dict else None
        self.s2 = batch_dict[S2_KEY] if S2_KEY in batch_dict else None
        self.s2_rir = batch_dict[S2_RIR] if S2_RIR in batch_dict else None
        self.anchor = batch_dict[ANCHOR_KEY] if ANCHOR_KEY in batch_dict else None
        self.n = batch_dict[NOISE_KEY] if NOISE_KEY in batch_dict else None
        self.n_rir = batch_dict[NOISE_RIR] if NOISE_RIR in batch_dict else None
        self.diffuse_noise = batch_dict[DIFFUSE_NOISE] if DIFFUSE_NOISE in batch_dict else None
        self.mix = batch_dict[MIX_KEY] if MIX_KEY in batch_dict else None
        
if __name__ == '__main__':
    from settings.config import TRAIN_CLEAN_SPEECH_PATH, TRAIN_NOISY_SPEECH_PATH, NOISE_PATH, TRAIN_MIC_RIR, TRAIN_DIFFUSE_NOISE, TRAIN_FRQ_RESPONSE, POINT_SNR_LIST, DIFFUSE_SNR_LIST, INTER_SNR_LIST
    s_mix_dataset = SECSMDataset(clean_s_dir=TRAIN_CLEAN_SPEECH_PATH,
                                    noisy_s_dir=TRAIN_NOISY_SPEECH_PATH,
                                    noise_dir=NOISE_PATH,
                                    rir_path=TRAIN_MIC_RIR,
                                    diffuse_noise_dir=TRAIN_DIFFUSE_NOISE)
    batch_dataloader = BatchDataLoader(s_mix_dataset, batch_size=4, 
                                    workers_num=0)
    data_factory = GPUDataSimulate(TRAIN_FRQ_RESPONSE, INTER_SNR_LIST, POINT_SNR_LIST, DIFFUSE_SNR_LIST, device='cpu')
    for batch_info in batch_dataloader:
        anchor, mix, s0_tgt, s1_tgt, s2_tgt, spk_id, spk_tgt, _, _ = data_factory(batch_info=batch_info)
        pass