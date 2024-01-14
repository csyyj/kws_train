import logging
import os
import random
import torch
import pickle
import numpy as np
import librosa
import torchaudio
import scipy.io as sio
import scipy.signal as signal
import soundfile as sf
import torch.nn as nn
from torch.autograd.variable import *
from abc import ABCMeta, abstractmethod
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tools.batch_rir_conv import batch_rir_conv, batch_rir_conv_same
from simulate_data.deploy_net_tcn_lstm_2mic import gen_net


SPEECH_KEY = 's'
SPEECH_TARGET_CHANNEL = 's_c'
ROAD_NOISE_KEY = 'road_noise'
S_RIR_KEY = 's_rir_key'
NFRAMES_KEY = 'nframes'

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


class CZDataset(Dataset):
    '''
    Basic class for genetating speech, mixture and vad info
    1. gen speech、mix
    2. intensify wav and noise
    3. vad info
    '''

    __metaclass__ = ABCMeta

    def __init__(self, wav_dir, noise_path, rir_dir, sample_rate=16000, speech_seconds=15, position_num=2):
        self.position_num = position_num
        self.wav_list = gen_target_file_list(wav_dir)
        self.road_noise_path_list = gen_target_file_list(noise_path, target_ext='.npy')
        self.rir_list = self._gen_rir_list(rir_dir)
        self.wav_len = 16000 * speech_seconds
        self.sample_rate = sample_rate
    

    def _gen_rir_list(self, rir_dict):
        p_rir = []
        for k, data in rir_dict.items():
            v = data['rir']
            channel = data['channel']
            rir_l = []
            if isinstance(v, list):
                for path in v:
                    rir_l += gen_target_file_list(path)
            else:
                rir_l += gen_target_file_list(v)
            p_rir.append((rir_l, int(channel)))
        return p_rir

    def __len__(self):
        return len(self.wav_list) * 10000

    def _choice_rir(self, num_person):
        position = random.sample(list(range(self.position_num)), num_person)
        rir_l = []
        channel_l = []
        for i in range(self.position_num):
            rir_pad = np.zeros([3000, self.position_num], dtype=np.float32)
            channel = 0
            if i in position:
                p_rir_l, channel = self.rir_list[i]
                sel_rir_path = p_rir_l[random.randint(0, len(p_rir_l) - 1)]
                rir, fs = sf.read(sel_rir_path)
                rir = rir / (np.max(np.abs(rir)) + 1e-4)
                rir_pad[:min(3000, rir.shape[0]), :rir.shape[1]] = rir[:min(3000, rir.shape[0]), :]
            rir_l.append(rir_pad.astype(np.float32).T)
            channel_l.append(channel)
        rir = np.stack(rir_l, axis=-1)[:rir.shape[1]]
        return position, rir, np.array(channel_l, dtype=np.int64)
    
        
    def _get_long_wav(self):
        path = self.wav_list[random.randint(0, len(self.wav_list) - 1)]
        wav, _ = sf.read(path)
        real_frames = wav.shape[0] // 256
        wav = np.concatenate([wav, np.zeros([self.wav_len - wav.shape[0]], dtype=np.float64)], axis=-1)
        return wav.astype(np.float32), real_frames


    def __getitem__(self, idx):
        num_person = random.randint(1, self.position_num)
        position, rirs, channel = self._choice_rir(num_person)
        s_l = []
        real_frames_l = []
        for i in range(self.position_num):
            if i in position:
                s_tmp, real_frames = self._get_long_wav()
            else:
                s_tmp = np.zeros([self.wav_len], dtype=np.float32)
                key_idx = 0
                label = np.array([-1], dtype=np.int64)
                real_frames = self.wav_len // 256
            s_l.append(s_tmp)
            real_frames_l.append(real_frames)

        s = np.stack(s_l, axis=1)
        real_frames = np.array(real_frames_l)

        road_noise_path = self.road_noise_path_list[random.randint(0, len(self.road_noise_path_list) - 1)]
        road_noise = np.load(road_noise_path, mmap_mode='c')
        noise_start = random.randint(0, road_noise.shape[1] - 1 - s.shape[0])
        sel_noise = road_noise[:, noise_start:noise_start + s.shape[0]]
        np.random.shuffle(sel_noise)
        sel_noise = sel_noise[:self.position_num]
        res_dict = {
            SPEECH_KEY: torch.from_numpy(s.astype(np.float32)),
            SPEECH_TARGET_CHANNEL: torch.from_numpy(channel),
            NFRAMES_KEY: torch.from_numpy(real_frames.astype(np.int64)),
            ROAD_NOISE_KEY: torch.from_numpy(sel_noise.astype(np.float32)),
            S_RIR_KEY: torch.from_numpy(rirs.astype(np.float32))
        }
        return res_dict
    

class BatchDataLoader(DataLoader):
    __metaclass__ = ABCMeta

    def __init__(self, s_mix_dataset, batch_size, is_shuffle=False, workers_num=0, sampler=None):
        super(BatchDataLoader, self).__init__(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                                  num_workers=workers_num, collate_fn=self.collate_fn, sampler=sampler)

    @staticmethod
    @abstractmethod
    def gen_batch_data(zip_res):
        return zip_res

    @staticmethod
    def collate_fn(batch):
        # batch.sort(key=lambda x: x[NFRAMES_KEY].max(), reverse=True)
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
            data = list(zip_res[i])
            if isinstance(data[0], torch.Tensor):
                data = pad_sequence(data, batch_first=True)
            batch_dict[key] = data
        # batch_dict = BatchDataLoader.gen_batch_data(batch_dict)
        return SMBatchInfo(batch_dict)
        


class GPUDataSimulate(nn.Module):
    def __init__(self, frq_response, road_snr_list, point_snr_list, device, zone_model_path):
        super(GPUDataSimulate, self).__init__()
        self.device = device
        self.road_snr_list = road_snr_list
        self.point_snr_list = point_snr_list
        self.hpf = self.gen_hpf()
        frq_response = np.load(frq_response, mmap_mode='c')
        self.frq_response = torch.from_numpy(frq_response.T)
        self.net = gen_net(zone_model_path)
        self.stft = STFT(512, 256)

    def __call__(self, s, n, s_rir):
        with torch.no_grad():
            mix, s = self.simulate_data(s, n, s_rir)
            s = s[:, :2]
            enhance_data, _, _ = self.net(mix[:, :2])
            b, c, t = enhance_data.size()
            s = s[:, :, :t]
            mix = mix[:, :, :t]
            enhance_data = enhance_data 
        return  enhance_data, s


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
        wav_max = torch.amax(wav.abs(), dim=1, keepdim=True)
        wav = batch_rir_conv_same(wav, torch.tile(self.hpf.unsqueeze(dim=0), [wav.size(0), 1]))
        if firs is None:
            idxs = np.random.choice(self.frq_response.shape[0], wav.size(0), replace=False)
            firs = self.frq_response[idxs]
        wav = batch_rir_conv_same(wav, firs)
        wav = wav / (torch.amax(wav.abs(), dim=1, keepdim=True) + 1e-5) * wav_max
        if is_need_fir:
            return wav, firs
        else:
            return wav
    
    def set_zeros(self, wavs):
        for i in range(wavs.size(0)):
            rdm_times = random.randint(1, 5)
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
        return wavs


    # @torch.compile
    def simulate_data(self, s, n, s_rir):
        '''
        s: [B, T, num_mic]
        n: [B, T, num_mic]
        rir: [B, T1, num_mic, num_person]
        '''
        with torch.no_grad():            
            s = s.to(self.device)
            n = n.to(self.device)      
            s_rir = s_rir.to(self.device)        

            s_rev_l = []
            s_tgt_l = []
            for i in range(s.size(-1)):
                s[..., i] = self.set_zeros(self.simulate_freq_response(s[..., i]))
                n[:, i] = self.simulate_freq_response(n[:, i])
                s_rev_tmp =  batch_rir_conv(s[..., i], s_rir[..., i])[..., :s.size(1)]
                s_tgt_tmp = s_rev_tmp[:, i, :]
                s_rev_l.append(s_rev_tmp)
                s_tgt_l.append(s_tgt_tmp)
            
            s_rev = torch.stack(s_rev_l, dim=-1).sum(-1)
            s_tgt = torch.stack(s_tgt_l, dim=1)

            s_power = (s_tgt ** 2).sum(-1) # [B, num_mic]
            not_zero_count = torch.where(s_tgt.abs() > 1e-4, torch.ones_like(s_tgt), torch.zeros_like(s_tgt)).sum(-1) + 1
            s_mean = s_power / not_zero_count
            s_max = torch.amax(s_mean)
            s_mean[s_mean < 1e-4] = s_max
            s_mean = torch.amin(s_mean, dim=-1)
            n_mean = (n ** 2).mean(-1)[:, 0]

            n_snr = torch.randint(self.road_snr_list[0], self.road_snr_list[-1], (s_mean.size(0),), dtype=s.dtype,
                                        device=s.device)
            n_alpha = (s_mean / (1e-7 + n_mean * (10.0 ** (n_snr / 10.0)))).sqrt().unsqueeze(dim=-1).unsqueeze(dim=-1)

            mix = s_rev + n[:, :s_rir.size(1)] * n_alpha
            mix_amp = torch.rand(s_rev.size(0), 1, 1, device=s_rev.device).clamp_(0.1, 1.0) 
            alpha = 1 / (torch.amax(torch.abs(mix), [-1, -2], keepdim=True) + 1e-7)
            alpha *= mix_amp
            mix *= alpha
            s_tgt *= alpha
            s_rev *= alpha
            return mix, s_tgt

class SMBatchInfo(object):
    def __init__(self, batch_dict):
        super(SMBatchInfo, self).__init__()
        self.s = batch_dict[SPEECH_KEY] if SPEECH_KEY in batch_dict else None
        self.real_frames = batch_dict[NFRAMES_KEY] if NFRAMES_KEY in batch_dict else None
        self.road_n = batch_dict[ROAD_NOISE_KEY] if ROAD_NOISE_KEY in batch_dict else None
        self.s_rir = batch_dict[S_RIR_KEY] if S_RIR_KEY in batch_dict else None


if __name__ == '__main__':
    WAV_PATH = '/home/yanyongjie/train_data/speech/TIMIT/data/lisa/data/timit'
    TRAINING_NOISE = '/home/yanyongjie/train_data/car_zone/noise/'
    TRAINING_RIR = {
                    'L1': {'rir': ['/home/yanyongjie/train_data/car_zone/rir/wulin/L1'],
                            'channel': 0},
                    'R1': {'rir': ['/home/yanyongjie/train_data/car_zone/rir/wulin/R1'],
                            'channel': 1},
                    'L2': {'rir': ['/home/yanyongjie/train_data/car_zone/rir/wulin/L2'],
                            'channel': 0},
                    'R2': {'rir': ['/home/yanyongjie/train_data/car_zone/rir/wulin/R2'],
                            'channel': 1},
                    }
    TRAIN_FRQ_RESPONSE = '/home/yanyongjie/train_data/fir/fir_1000000.npy'
    dataset = CZDataset(wav_dir=WAV_PATH, noise_path=TRAINING_NOISE, rir_dir=TRAINING_RIR)
    ROAD_SNR_LIST = [-5, -2, -1, 0, 1, 2, 5]
    POINT_SNR_LIST = [2, 3, 5, 10]
    batch_dataloader = BatchDataLoader(dataset, batch_size=4, workers_num=0)
    car_zone_model_path = '/home/yanyongjie/code/official/car/car_zone_wuling_no_back_r4/model/model-302000--20.740382461547853.pickle'
    data_factory = GPUDataSimulate(TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_SNR_LIST, device='cpu', zone_model_path=car_zone_model_path)
    for batch_info in batch_dataloader:
        res = data_factory(batch_info.s, batch_info.road_n, batch_info.s_rir)
        pass

