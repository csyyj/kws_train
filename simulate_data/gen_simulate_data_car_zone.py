import logging
import os
import random
import torch
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import soundfile as sf
import torch.nn as nn
from torch.autograd.variable import *
from abc import ABCMeta, abstractmethod
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tools.batch_rir_conv import batch_rir_conv, batch_rir_conv_same
from tools.torch_stft import STFT
from settings.config import TRAINING_NOISE, TRAINING_RIR, TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_NOISE_PATH, POINT_SNR_LIST
from simulate_data.deploy_net_tcn_lstm_2mic import gen_net

EPSILON = 1e-7
NOISE_PARTS_NUM = 20

#
SPEECH_KEY = 's'
KEY_WORDS_LEBEL = 'label_idx'
ROAD_NOISE_KEY = 'road_noise'  # 原始的mic信号，纬度[mic_num, t, mic_num], S_ORI[i]表示第i个位置说话人到每个mic的信号
P_NOISE_KEY = 'p_noise'
S_RIR_KEY = 's_rir_key'
P_NOISE_RIR_KEY = 'p_noise_rir_key'

# dataset config
# tuning
SPEECH_ONLY_RATE = 0.01  # 0.01
NOISE_ONLY_RATE = 0.01  # 0.01
CLIP_RATE = 0.01
RIR_RATE = 0.0

# float energy
SPEECH_MAX = 0.95
SPEECH_MIN = 0.05


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

    def __init__(self, kws_wav_dir, bg_wav_dir, noise_path, rir_dir, p_noise_dir, sample_rate=16000, speech_seconds=10, mic_num=4):
        self.key_words_list = self.gen_kws_list(kws_wav_dir)
        self.bg_list = self.gen_speaker_list(bg_wav_dir)
        self.mic_num = mic_num
        self.road_noise_path_list = gen_target_file_list(noise_path, target_ext='.npy')
        self.noise_data_info = self._list_noise_and_snr(p_noise_dir)
        self.rir_list = self._gen_rir_list(rir_dir)
        self.wav_len = 16000 * speech_seconds
        self.sample_rate = sample_rate
    
    def gen_kws_list(self, kws_wav_dir):
        keys = []
        for path in kws_wav_dir:
            tmp_l = gen_target_file_list(path)
            keys.append(tmp_l)
        return keys
        

    def _gen_rir_list(self, rir_dict):
        p_rir = []
        for k, v in rir_dict.items():
            rir_l = []
            if isinstance(v, list):
                for path in v:
                    rir_l += gen_target_file_list(path)
            else:
                rir_l += gen_target_file_list(v)
            p_rir.append(rir_l)
        return p_rir
    
    def _list_noise_and_snr(self, noise_path):
        noise_bin_ext = '.bin'
        noise_info = []
        for f in os.listdir(noise_path):
            full_path = os.path.join(noise_path, f)
            if os.path.isfile(full_path) and os.path.splitext(f)[1] == noise_bin_ext:
                noise_np = np.memmap(full_path, dtype='float32', mode='c')
                noise_np = noise_np[:noise_np.shape[0] // NOISE_PARTS_NUM * NOISE_PARTS_NUM]
                noise_data = np.reshape(noise_np, (NOISE_PARTS_NUM, noise_np.shape[0] // NOISE_PARTS_NUM))
                noise_info.append(noise_data)
        return noise_info

    def gen_speaker_list(self, speech_dir):
        wav_l = []
        for root, dirs, files in os.walk(speech_dir, followlinks=True):
            for f in files:
                f = os.path.join(root, f)
                ext = os.path.splitext(f)[1]
                ext = ext.lower()
                if ext == '.npy' and '._' not in f and 'cat_speaker' in f and ('elevoc_vad' not in f):
                    wav_l.append(f)
        return wav_l

    def _gen_filter(self, filter_path):
        filter_path = gen_target_file_list(filter_path, '.mat')
        filter_list = []
        for path in filter_path:
            p_filter = sio.loadmat(path)
            filter_list.append(p_filter)
        return filter_list

    def __len__(self):
        return len(self.bg_list) * 10000

    def _set_zero(self, wav1):
        rdm_times = random.randint(1, 3)
        rdm_index_l = []
        for _ in range(rdm_times):
            rdm_index_l.append(random.randint(0, wav1.size - 16000))
        start_index_l = sorted(rdm_index_l)
        start_index_l.append(wav1.size)
        for i in range(len(start_index_l) - 1):
            start = start_index_l[i]
            end = random.randint(start, start_index_l[i + 1])
            wav1[start:end] = 0
        return wav1

    def _read_road_noise(self):
        pass

    def _gen_mic_data(self, speech, trans_func):
        s_rir = signal.fftconvolve(speech.reshape(-1, 1), trans_func, mode='full')[:speech.shape[0]]
        for i in range(s_rir.shape[1]):
            s_rir[:, i] = self._simulate_freq_response(s_rir[:, i])
        return s_rir

    def _choice_rir(self, num_person):
        position = random.sample(list(range(self.mic_num)), num_person)
        rir_l = []
        for i in range(self.mic_num):
            rir_pad = np.zeros([3000, self.mic_num], dtype=np.float32)
            if i in position:
                p_rir_l = self.rir_list[i]
                sel_rir_path = p_rir_l[random.randint(0, len(p_rir_l) - 1)]
                rir, fs = sf.read(sel_rir_path)
                rir = rir / (np.max(np.abs(rir)) + 1e-4)
                rir_pad[:min(3000, rir.shape[0]), :] = rir[:min(3000, rir.shape[0]), :self.mic_num]
            rir_l.append(rir_pad.astype(np.float32).T)
        rir = np.stack(rir_l, axis=-1)
        return position, rir
    
    def _read_point_noise(self, noise_len=None):
        if noise_len is None:
            noise_len = self.wav_len
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

    def __getitem__(self, idx):
        num_person = random.randint(1, self.mic_num)
        position, rirs = self._choice_rir(num_person)
        s_l = []
        key_idx_l = []
        for i in range(self.mic_num):
            if i in position:
                if i < 2:
                    s_tmp, key_idx = self._get_long_wav(is_key=False)
                else:
                    s_tmp, key_idx = self._get_long_wav()
            else:
                s_tmp = np.zeros([self.wav_len], dtype=np.float32)
                key_idx = 0
            if key_idx > 0:
                amp = random.uniform(0.5, 0.8)
            else:
                amp = random.uniform(0.3, 0.5)
            s_tmp = s_tmp / (np.max(np.abs(s_tmp)) + 1e-6) * amp
            s_l.append(s_tmp)
            key_idx_l.append(key_idx)
        s = np.stack(s_l, axis=1)
        key_idx = np.array(key_idx_l, dtype=np.int64)
        
        num_p_noise = random.randint(1, max(2, self.mic_num // 2 + 1))
        p_position, p_rirs = self._choice_rir(num_p_noise)
        n_l = []
        for i in range(self.mic_num):
            if i in p_position:
                p_n_tmp = self._read_point_noise()
            else:
                p_n_tmp = np.zeros([self.wav_len], dtype=np.float32)
            n_l.append(p_n_tmp)
        p_noise = np.stack(n_l, axis=1)
        road_noise_path = self.road_noise_path_list[random.randint(0, len(self.road_noise_path_list) - 1)]
        road_noise = np.load(road_noise_path, mmap_mode='c')
        noise_start = random.randint(0, road_noise.shape[1] - 1 - s.shape[0])
        sel_noise = road_noise[:, noise_start:noise_start + s.shape[0]]
        np.random.shuffle(sel_noise)
        sel_noise = sel_noise[:self.mic_num]
        res_dict = {
            SPEECH_KEY: torch.from_numpy(s.astype(np.float32)),
            KEY_WORDS_LEBEL: torch.from_numpy(key_idx.astype(np.int64)),
            P_NOISE_KEY: torch.from_numpy(p_noise.astype(np.float32)),
            ROAD_NOISE_KEY: torch.from_numpy(sel_noise.astype(np.float32)),
            S_RIR_KEY: torch.from_numpy(rirs.astype(np.float32)),
            P_NOISE_RIR_KEY: torch.from_numpy(p_rirs.astype(np.float32))
        }
        return res_dict

    def _simulate_freq_response(self, wav):
        b = np.array([1.0, -2, 1])
        a = np.array([1.0, -1.99599, 0.99600])
        wav = signal.lfilter(b, a, wav)

        r1, r2, r3, r4 = (random.uniform(-3 / 8, 3 / 8) for i in range(4))
        b = np.array([1.0, r1, r2])
        a = np.array([1.0, r3, r4])
        wav = signal.lfilter(b, a, wav)
        return wav.astype(np.float32)

    def _simulate_multi_mic_freq_response(self, wav):
        '''
            模拟mic freq response
        :param wav: 1_d numpy or 2_d numpy which shape is [t, c]
        :return:
        '''
        if len(wav.shape) == 1:
            return self._simulate_freq_response(wav)
        else:
            l = []
            for i in range(wav.shape[-1]):
                l.append(self._simulate_freq_response(wav[:, i]))
            return np.stack(l, axis=-1)

    def _get_long_wav(self, is_key=None):
        if is_key is None:
            if random.random() < 0.5:
                is_key = True
            else:
                is_key = False
        if is_key:
            # key word
            while True:
                idx = random.randint(0, len(self.key_words_list) - 1)
                key_path_list = self.key_words_list[idx]
                key_path = key_path_list[random.randint(0, len(key_path_list) - 1)]
                wav, _ = sf.read(key_path)
                wav = wav / (np.max(np.abs(wav)) + 1e-6)
                if wav.shape[0] < self.wav_len:
                    if random.random() < 0:
                        key_path2 = key_path_list[random.randint(0, len(key_path_list) - 1)]
                        wav2, _ = sf.read(key_path2)
                        wav2 = wav2 / (np.max(np.abs(wav2)) + 1e-6)
                        if wav2.shape[0] + wav.shape[0] <= self.wav_len:
                            wav = np.concatenate([wav, wav2], axis=0)
                    left_pad = random.randint(0, self.wav_len - wav.shape[0])
                    wav = np.pad(wav, [left_pad, self.wav_len - wav.shape[0] - left_pad])
                    idx = idx + 1
                    break
                else:
                    continue
        else:
            # background
            while True:
                spk_id = random.randint(0, len(self.bg_list) - 1)
                spk_path = self.bg_list[spk_id]
                spk_wav = np.load(spk_path, mmap_mode='c')
                if spk_wav.size - 1 - self.wav_len > 0:
                    break
            rdm_start = random.randint(0, spk_wav.size - 1 - self.wav_len)
            wav = spk_wav[rdm_start:rdm_start + self.wav_len]
            idx = 0
        return wav.astype(np.float32), idx


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

    def __call__(self, batch_info):
        s, n, p_n, s_rir, p_rir, label_idx = batch_info.s, batch_info.road_n, batch_info.p_n, batch_info.s_rir, batch_info.p_rir, batch_info.label_idx
        mix, s = self.simulate_data(s, n, p_n, s_rir, p_rir)
        enhance_data, _, _ = self.net(mix[:, 2:])
        b, c, _ = enhance_data.size()
        enhance_data = enhance_data + random.uniform(0.01, 0.2) * self.stft(s[:, 2:].reshape(b * c, -1)).reshape(b, c, -1)
        b, c, t = enhance_data.shape
        enhance_l = []
        s_l = []
        label_idx_l = []
        for i in range(enhance_data.size(0)):
            if label_idx[i].sum().item() > 0:
                if label_idx[i, 2].item() > 0:
                    enhance_l.append(enhance_data[i, 0])
                    s_l.append(s[i, 2])
                    label_idx_l.append(label_idx[i, 2])
                if label_idx[i, 3].item() > 0:
                    enhance_l.append(enhance_data[i, 1])
                    s_l.append(s[i, 3])
                    label_idx_l.append(label_idx[i, 3])
            else:
                enhance_l.append(enhance_data[i, 0])
                s_l.append(s[i, 2])
                label_idx_l.append(label_idx[i, 2])
                enhance_l.append(enhance_data[i, 1])
                s_l.append(s[i, 3])
                label_idx_l.append(label_idx[i, 3])
        enhance_data = torch.stack(enhance_l, dim=0)
        s = torch.stack(s_l, dim=0)
        label_idx = torch.stack(label_idx_l, dim=0)
        return  enhance_data, s, label_idx


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
    def simulate_data(self, s, n, p_n, s_rir, p_rir):
        '''
        s: [B, T, num_mic]
        n: [B, T, num_mic]
        rir: [B, T1, num_mic, num_person]
        '''
        with torch.no_grad():
            s = s.to(self.device)
            n = n.to(self.device)
            p_n = p_n.to(self.device)
            s_rir = s_rir.to(self.device)            
            p_rir = p_rir.to(self.device) 

            s_rev_l = []
            p_rev_l = []
            s_tgt_l = []
            for i in range(s.size(-1)):
                s[..., i] = self.simulate_freq_response(s[..., i])
                n[:, i] = self.simulate_freq_response(n[:, i])
                p_n[..., i] = self.simulate_freq_response(p_n[..., i])
                s_rev_tmp =  batch_rir_conv(s[..., i], s_rir[..., i])[..., :s.size(1)]
                p_n_rev_tmp =  batch_rir_conv(p_n[..., i], p_rir[..., i])[..., :s.size(1)]
                s_tgt_tmp = s_rev_tmp[:, i]
                s_rev_l.append(s_rev_tmp)
                s_tgt_l.append(s_tgt_tmp)
                p_rev_l.append(p_n_rev_tmp)
            
            s_rev = torch.stack(s_rev_l, dim=-1).sum(-1)
            p_rev = torch.stack(p_rev_l, dim=-1).sum(-1)
            s_tgt = torch.stack(s_tgt_l, dim=1)

            s_power = (s_tgt ** 2).sum(-1) # [B, num_mic]
            not_zero_count = torch.where(s_tgt.abs() > 1e-4, torch.ones_like(s_tgt), torch.zeros_like(s_tgt)).sum(-1) + 1
            s_mean = s_power / not_zero_count
            s_max = torch.amax(s_mean)
            s_mean[s_mean < 1e-4] = s_max
            s_mean = torch.amin(s_mean, dim=-1)
            n_mean = (n ** 2).mean(-1)[:, 0]
            p_n_mean = torch.amax((p_rev ** 2).mean(-1), dim=-1)

            n_snr = torch.randint(self.road_snr_list[0], self.road_snr_list[-1], (s_mean.size(0),), dtype=s.dtype,
                                        device=s.device)
            n_alpha = (s_mean / (1e-7 + n_mean * (10.0 ** (n_snr / 10.0)))).sqrt().unsqueeze(dim=-1).unsqueeze(dim=-1)
            
            p_n_snr = torch.randint(self.point_snr_list[0], self.point_snr_list[-1], (s_mean.size(0),), dtype=s.dtype,
                                        device=s.device)
            p_n_alpha = (s_mean / (1e-7 + p_n_mean * (10.0 ** (p_n_snr / 10.0)))).sqrt().unsqueeze(dim=-1).unsqueeze(dim=-1)

            mix = s_rev + n * n_alpha + p_rev * p_n_alpha
            mix_amp = torch.rand(s_rev.size(0), 1, 1, device=s_rev.device).clamp_(0.1, 1.0)
            alpha = 1 / (torch.amax(torch.abs(mix), [-1, -2], keepdim=True) + EPSILON) * mix_amp
            mix *= alpha
            s_tgt *= alpha
            s_rev *= alpha
            return mix, s_tgt




class SMBatchInfo(object):
    def __init__(self, batch_dict):
        super(SMBatchInfo, self).__init__()
        self.s = batch_dict[SPEECH_KEY] if SPEECH_KEY in batch_dict else None
        self.label_idx = batch_dict[KEY_WORDS_LEBEL] if KEY_WORDS_LEBEL in batch_dict else None
        self.road_n = batch_dict[ROAD_NOISE_KEY] if ROAD_NOISE_KEY in batch_dict else None
        self.p_n = batch_dict[P_NOISE_KEY] if P_NOISE_KEY in batch_dict else None
        self.s_rir = batch_dict[S_RIR_KEY] if S_RIR_KEY in batch_dict else None
        self.p_rir = batch_dict[P_NOISE_RIR_KEY] if P_NOISE_RIR_KEY in batch_dict else None


if __name__ == '__main__':
    from settings.config import *
    dataset = CZDataset(TRAINING_KEY_WORDS, TRAINING_BACKGROUND, TRAINING_NOISE, TRAINING_RIR, POINT_NOISE_PATH,
                        sample_rate=16000, speech_seconds=15)
    batch_dataloader = BatchDataLoader(dataset, batch_size=4, workers_num=0)
    car_zone_model_path = '/home/yanyongjie/code/official/car/car_zone_2_for_aodi_real/model/student_model/model-1200000--17.81806887626648.pickle'
    data_factory = GPUDataSimulate(TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_SNR_LIST, device='cpu', zone_model_path=car_zone_model_path)
    for batch_info in batch_dataloader:
        res = data_factory(batch_info)
        pass
