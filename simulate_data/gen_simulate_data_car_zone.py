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
from tools.torch_stft import STFT
from settings.config import TRAINING_NOISE, TRAINING_RIR, TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_NOISE_PATH, POINT_SNR_LIST
from simulate_data.deploy_net_tcn_lstm_2mic import gen_net

EPSILON = 1e-7
NOISE_PARTS_NUM = 20

#
SPEECH_KEY = 's'
SPEECH_TARGET_CHANNEL = 's_c'
KEY_WORDS_LEBEL = 'label_idx'
CUSTOM_LABEL = 'custom_label'
CUSTOM_LABEL_LEN = 'custom_label_len'
ROAD_NOISE_KEY = 'road_noise'  # 原始的mic信号，纬度[mic_num, t, mic_num], S_ORI[i]表示第i个位置说话人到每个mic的信号
P_NOISE_KEY = 'p_noise'
S_RIR_KEY = 's_rir_key'
P_NOISE_RIR_KEY = 'p_noise_rir_key'
NFRAMES_KEY = 'nframes'
LABEL_FRAMES_KEY = 'label_frames'

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

    def __init__(self, pin_yin_config_path, kws_wav_dir, bg_wav_dir, noise_path, rir_dir, p_noise_dir, error_kws_dir, sample_rate=16000, speech_seconds=15, position_num=2):
        self.error_kws_list = gen_target_file_list(error_kws_dir)
        self.position_num = position_num
        self.pin_yin_config = self.parse_pin_yin_config(pin_yin_config_path)
        self.key_words_list = self.gen_kw_pickle_list(kws_wav_dir)
        self.bg_wav_list = self.gen_pickle_list(bg_wav_dir)
        self.road_noise_path_list = gen_target_file_list(noise_path, target_ext='.npy')
        self.noise_data_info = self._list_noise_and_snr(p_noise_dir)
        self.rir_list = self._gen_rir_list(rir_dir)
        self.wav_len = 16000 * speech_seconds
        self.sample_rate = sample_rate
    
    def parse_pin_yin_config(self, pin_yin_config_path):
        with open(pin_yin_config_path, 'r') as f:
            data = f.readline()
            dict = {}
            while data:
                key, value = data.split('\t')
                dict[key] = value.replace('\n', '')
                data = f.readline()
        return dict
            
    
    def gen_pickle_list(self, pickle_pathes):
        keys = []
        for path in pickle_pathes:
            with open(path, 'rb') as f:
                tmp = pickle.load(f)
            keys += tmp
        return keys
    
    def gen_kw_pickle_list(self, pickle_pathes):
        keys = []
        for info in pickle_pathes:
            if isinstance(info, list):
                tmp = []
                for p in info:
                    if isinstance(p, tuple):
                        scale = p[1]
                        p = p[0]
                    else:
                        scale = 1
                    with open(p, 'rb') as f:
                        key_tmp = pickle.load(f)
                    tmp += key_tmp * scale
            else:
                with open(info, 'rb') as f:
                    tmp = pickle.load(f)
            keys.append(tmp)
        return keys
        

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
        return len(self.bg_wav_list) * 10000

    def _read_road_noise(self):
        pass

    def _gen_mic_data(self, speech, trans_func):
        s_rir = signal.fftconvolve(speech.reshape(-1, 1), trans_func, mode='full')[:speech.shape[0]]
        for i in range(s_rir.shape[1]):
            s_rir[:, i] = self._simulate_freq_response(s_rir[:, i])
        return s_rir

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
        num_person = random.randint(1, self.position_num)
        position, rirs, channel = self._choice_rir(num_person)
        s_l = []
        key_idx_l = []
        max_label_len = 0
        label_l = []
        label_len_l = []
        real_frames_l = []
        label_frame_l = []
        is_has_key = False
        for i in range(self.position_num):
            if i in position:
                if not is_has_key:
                    # if random.random() < (1 / (len(self.key_words_list) + 1)):
                    if random.random() < 0.6 or i > 1:
                        is_key = False
                    else:
                        is_key = True
                else:
                    is_key = False
                
                if random.random() < 0.95:
                    s_tmp, key_idx, label, real_frames, label_frame = self._get_long_wav(is_key=is_key)
                else:
                    s_tmp, key_idx, label, real_frames, label_frame = self._get_error_kws_wav()
                
                if key_idx > 0:
                    is_has_key = True
                
                # s_tmp = self._simulate_freq_response(s_tmp)
                label_len_l.append(label.size)
            else:
                s_tmp = np.zeros([self.wav_len], dtype=np.float32)
                key_idx = 0
                label = np.array([-1], dtype=np.int64)
                real_frames = self.wav_len // 256
                label_len_l.append(0)
                label_frame = -1
            if max_label_len < label.shape[0]:
                max_label_len = label.shape[0]
            if key_idx > 0:
                amp = random.uniform(0.5, 0.8)
            else:
                amp = random.uniform(0.3, 0.5)
            s_tmp = s_tmp / (np.max(np.abs(s_tmp)) + 1e-6) * amp
            s_l.append(s_tmp)
            key_idx_l.append(key_idx)
            label_l.append(label)
            real_frames_l.append(real_frames)
            label_frame_l.append(label_frame)
        
        # s_l_new = []
        label_l_new = []
        for i in range(self.position_num):
            # s_tmp = s_l[i]
            # s_l_new.append(np.pad(s_tmp, mode='constant', pad_width=[0, max_len - s_tmp.size], constant_values=[0, 0]))
            label_tmp = label_l[i]
            label_l_new.append(np.pad(label_tmp, mode='constant', pad_width=[0, max_label_len - label_tmp.size], constant_values=[0, 0]))
        
        s = np.stack(s_l, axis=1)
        label = np.stack(label_l_new, axis=1)
        label_len = np.array(label_len_l, dtype=np.int64)
        key_idx = np.array(key_idx_l, dtype=np.int64)
        real_frames = np.array(real_frames_l)
        label_frames = np.array(label_frame_l)
        
        num_p_noise = random.randint(1, max(2, self.position_num // 2 + 1))
        p_position, p_rirs, _ = self._choice_rir(num_p_noise)
        n_l = []
        for i in range(self.position_num):
            if i in p_position:
                p_n_tmp = self._read_point_noise(self.wav_len)
            else:
                p_n_tmp = np.zeros([self.wav_len], dtype=np.float32)
            n_l.append(p_n_tmp)
        p_noise = np.stack(n_l, axis=1)
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
            LABEL_FRAMES_KEY: torch.from_numpy(label_frames.astype(np.int64)),
            KEY_WORDS_LEBEL: torch.from_numpy(key_idx.astype(np.int64)),
            CUSTOM_LABEL: torch.from_numpy(label.astype(np.int64)),
            CUSTOM_LABEL_LEN: torch.from_numpy(label_len.astype(np.int64)),
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
    
    def gen_label_wav(self, info):
        path = os.path.join('/mnt/raid2/user_space/yanyongjie/asr', info[1])
        if not os.path.exists(path):
            print('wav not find: {}'.format(path))
            return None, None, False, None
        npy_path = path.replace('.wav', '.npy')
        label_path = path.replace('.wav', '.txt')
        if os.path.exists(npy_path):
            try:
                data = np.load(npy_path, mmap_mode='c')
            except:
                data, fs = sf.read(path)
                if fs != 16000:
                    data = librosa.resample(data, orig_sr=fs, target_sr=16000)
                np.save(npy_path, data.astype(np.float16))
        else:
            try:
                data, fs = sf.read(path)
            except:
                print(path)
                return None, None, False, None
            if fs != 16000:
                data = librosa.resample(data, orig_sr=fs, target_sr=16000)
            np.save(npy_path, data.astype(np.float16))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                try:
                    label_sample = int(f.read())
                    label_frame = label_sample // (16 * 16)
                except:
                    print('{} read error'.format(label_sample))
                    return None, None, False, None
        else:
            label_frame = -1
            if random.random() < 0.3:
                # 变速
                t = data.shape
                sox_rate = random.uniform(0.9, 1.1)
                s_tmp_tensor = torch.from_numpy(data.astype(np.float32)).reshape(1, -1)
                data, _ = torchaudio.sox_effects.apply_effects_tensor(
                    s_tmp_tensor, 16000, [['speed', str(sox_rate)], ['rate', str(16000)]]
                    )
                data = data.squeeze().cpu().numpy()
        label = self.pinyin2idx(info[2], info)
        return data, label, True, label_frame

    def pinyin2idx(self, pin_yin, info):
        slice_in = pin_yin.split(' ')
        l = []
        for p in slice_in:
            try:
                idx = self.pin_yin_config[p]
                l.append(idx)
            except:
                print(info)
        return np.array(l, dtype=np.int64)
    
    def _get_long_wav(self, is_key):
        if is_key:
            # key word
            while True:
                idx = random.randint(0, len(self.key_words_list) - 1)
                if random.random() < 0.5:
                    idx = random.randint(0, len(self.key_words_list) - 1)
                else:
                    idx = 0
                key_list = self.key_words_list[idx]
                rdm_idx = random.randint(0, len(key_list) - 1)
                bg_info = key_list[rdm_idx]
                wav, label, success, label_frame = self.gen_label_wav(bg_info)
                if not success:
                    continue
                wav = wav / (np.max(np.abs(wav)) + 1e-6)
                if wav.shape[0] < self.wav_len:
                    break
                else:
                    continue
            idx = idx + 1
        else:
            # background
            while True:
                idx = random.randint(0, len(self.bg_wav_list) - 1)
                bg_info = self.bg_wav_list[idx]
                wav, label, success, label_frame = self.gen_label_wav(bg_info)
                if not success:
                    continue
                wav = wav / (np.max(np.abs(wav)) + 1e-6)
                if wav.shape[0] < self.wav_len:
                    break
                else:
                    continue
            idx = 0
        real_frames = wav.shape[0] // 256
        wav = np.concatenate([wav, np.zeros([self.wav_len - wav.shape[0]], dtype=np.float64)], axis=-1)
        return wav.astype(np.float32), idx, label, real_frames, label_frame
    
    def _get_error_kws_wav(self):
        while True:
            error_kws_path = self.error_kws_list[random.randint(0, len(self.error_kws_list) - 1)]
            wav, _ = sf.read(error_kws_path)
            if wav.shape[0] > 16000:
                break
        idx = 0
        label = np.array([-1], np.int64)
        real_frames = wav.shape[0] // 256
        wav = np.concatenate([wav, np.zeros([self.wav_len - wav.shape[0]], dtype=np.float64)], axis=-1)
        label_frame = -1
        return wav.astype(np.float32), idx, label, real_frames, label_frame
        
        


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

    def __call__(self, batch_info):
        with torch.no_grad():
            s, n, p_n, s_rir, channel, p_rir, label_idx, custom_label, custom_label_len, real_frames, label_frames = \
                batch_info.s, batch_info.road_n, batch_info.p_n, batch_info.s_rir, batch_info.channel, batch_info.p_rir, batch_info.label_idx, batch_info.custom_label, batch_info.custom_label_len, batch_info.real_frames, batch_info.label_frames
            mix, s, mix_no_inter, label_idx, custom_label, custom_label_len, real_frames, label_frames = \
                self.simulate_data(s, n, p_n, s_rir, channel, p_rir, label_idx, custom_label, custom_label_len, real_frames, label_frames)
            s = s[:, :2]
            enhance_data, _, _ = self.net(mix[:, :2])
            b, c, t = enhance_data.size()
            s = s[:, :, :t]
            mix = mix[:, :, :t]
            mix_no_inter = mix_no_inter[:, :, :t]
            enhance_data = enhance_data #+ random.uniform(0.1, 0.3) * self.stft(s.reshape(b * c, -1)).reshape(b, c, -1)
            # 先用 clean 训练看
            # enhance_data = self.stft(s[:, 2:].reshape(b * c, -1)).reshape(b, c, -1)
            b, c, t = enhance_data.shape
            enhance_l = []
            s_l = []
            label_idx_l = []
            custom_label_l = []
            real_frames_l = []
            label_frames_l = []
            custom_label_len_l = []
            for i in range(enhance_data.size(0)):
                if label_idx[i].sum().item() > 0:
                    if label_idx[i, 0].item() > 0:
                        rdm_rate = random.random()
                        if rdm_rate < 0.7:
                            enhance_l.append(enhance_data[i, 0])
                        elif rdm_rate < 0.8:
                            enhance_l.append(mix_no_inter[i, 0])
                        elif rdm_rate < 0.85:
                            enhance_l.append(mix[i, 0])
                        else:
                            enhance_l.append(s[i, 0])
                        s_l.append(s[i, 0])
                        label_idx_l.append(label_idx[i, 0])
                        custom_label_l.append(custom_label[i, :, 0])
                        real_frames_l.append(real_frames[i, 0])
                        label_frames_l.append(label_frames[i, 0])
                        custom_label_len_l.append(custom_label_len[i, 0])
                    if label_idx[i, 1].item() > 0:
                        rdm_rate = random.random()
                        if rdm_rate < 0.7:
                            enhance_l.append(enhance_data[i, 1])
                        elif rdm_rate < 0.8:
                            enhance_l.append(mix_no_inter[i, 1])
                        elif rdm_rate < 0.85:
                            enhance_l.append(mix[i, 1])
                        else:
                            enhance_l.append(s[i, 1])
                        s_l.append(s[i, 1])
                        label_idx_l.append(label_idx[i, 1])
                        custom_label_l.append(custom_label[i, :, 1])
                        real_frames_l.append(real_frames[i, 1])
                        label_frames_l.append(label_frames[i, 1])
                        custom_label_len_l.append(custom_label_len[i, 1])
                else:
                    rdm_rate = random.random()
                    if rdm_rate < 0.7:
                        enhance_l.append(enhance_data[i, 0])
                    elif rdm_rate < 0.8:
                        enhance_l.append(mix_no_inter[i, 0])
                    elif rdm_rate < 0.85:
                        enhance_l.append(mix[i, 0])
                    else:
                        enhance_l.append(s[i, 0])
                    s_l.append(s[i, 0])
                    label_idx_l.append(label_idx[i, 0])
                    custom_label_l.append(custom_label[i, :, 0])
                    real_frames_l.append(real_frames[i, 0])
                    label_frames_l.append(label_frames[i, 0])
                    custom_label_len_l.append(custom_label_len[i, 0])
                    
                    rdm_rate = random.random()
                    if rdm_rate < 0.7:
                        enhance_l.append(enhance_data[i, 1])
                    elif rdm_rate < 0.8:
                        enhance_l.append(mix_no_inter[i, 1])
                    elif rdm_rate < 0.85:
                        enhance_l.append(mix[i, 1])
                    else:
                        enhance_l.append(s[i, 1])
                    s_l.append(s[i, 1])
                    label_idx_l.append(label_idx[i, 1])
                    custom_label_l.append(custom_label[i, :, 1])
                    real_frames_l.append(real_frames[i, 1])
                    label_frames_l.append(label_frames[i, 1])
                    custom_label_len_l.append(custom_label_len[i, 1])
            enhance_data = torch.stack(enhance_l, dim=0)
            enhance_data, firs = self.simulate_freq_response(enhance_data, is_need_fir=True)
            s = torch.stack(s_l, dim=0)
            s = self.simulate_freq_response(s, firs=firs)
            label_idx = torch.stack(label_idx_l, dim=0)
            custom_label = torch.stack(custom_label_l, dim=0)
            real_frames = torch.stack(real_frames_l, dim=0)
            label_frames = torch.stack(label_frames_l, dim=0)
            custom_label_len = torch.stack(custom_label_len_l, dim=0)
        return  enhance_data, s, label_idx, custom_label, custom_label_len, real_frames, label_frames


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
    def simulate_data(self, s, n, p_n, s_rir, channel, p_rir, label_idx, custom_label, custom_label_len, real_frames, label_frames):
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
            channel = channel.to(self.device)          
            p_rir = p_rir.to(self.device) 

            s_rev_l = []
            p_rev_l = []
            s_tgt_l = []
            for i in range(s.size(-1)):
                s[..., i] = self.set_zeros(self.simulate_freq_response(s[..., i]))
                n[:, i] = self.simulate_freq_response(n[:, i])
                p_n[..., i] = self.simulate_freq_response(p_n[..., i])
                s_rev_tmp =  batch_rir_conv(s[..., i], s_rir[..., i])[..., :s.size(1)]
                p_n_rev_tmp =  batch_rir_conv(p_n[..., i], p_rir[..., i])[..., :s.size(1)]
                s_tgt_tmp = s_rev_tmp[torch.arange(s.size(0)), channel[:, i], :]
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
            
            for i in range(s_rev.size(1)):
                #label_idx, custom_label, custom_label_len, real_frames, label_frames
                if random.random() < 0.3:
                    s_rev[i] = 0
                    s_tgt[i] = 0
                    label_idx[i] = 0
                    custom_label[i] = -1
                    label_frames[i] = -1

            mix = s_rev + n[:, :s_rir.size(1)] * n_alpha + p_rev * p_n_alpha
            mix_no_inter = s_tgt[:, :s_rev.size(1)] + n[:, :s_rir.size(1)] * n_alpha + p_rev * p_n_alpha
            mix_amp = torch.rand(s_rev.size(0), 1, 1, device=s_rev.device).clamp_(0.1, 1.0) 
            alpha = 1 / (torch.amax(torch.abs(mix), [-1, -2], keepdim=True) + EPSILON)
            alpha *= mix_amp
            mix_no_inter *= alpha
            mix *= alpha
            s_tgt *= alpha
            s_rev *= alpha
            # for i in range(mix.size(0)):
            #     if random.random() < 0.3:
            #         mix[i] = mix[i] / (torch.amax(torch.abs(mix[i]), keepdim=True) + 1e-6)
            #         amp = random.uniform(0.03, 0.7)
            #         mix[i] = torch.clamp(mix[i], -amp,  amp)
                    
            #         mix_no_inter[i] = mix_no_inter[i] / (torch.amax(torch.abs(mix_no_inter[i]), keepdim=True) + 1e-6)
            #         mix_no_inter[i] = torch.clamp(mix_no_inter[i], -amp,  amp)
                    
            #         s_tgt[i] = s_tgt[i] / (torch.amax(torch.abs(s_tgt[i]), keepdim=True) + 1e-6)
            #         s_tgt[i] = torch.clamp(s_tgt[i], -amp,  amp)
            return mix, s_tgt, mix_no_inter, label_idx, custom_label, custom_label_len, real_frames, label_frames




class SMBatchInfo(object):
    def __init__(self, batch_dict):
        super(SMBatchInfo, self).__init__()
        self.s = batch_dict[SPEECH_KEY] if SPEECH_KEY in batch_dict else None
        self.channel = batch_dict[SPEECH_TARGET_CHANNEL] if SPEECH_TARGET_CHANNEL in batch_dict else None
        self.label_frames = batch_dict[LABEL_FRAMES_KEY] if LABEL_FRAMES_KEY in batch_dict else None
        self.real_frames = batch_dict[NFRAMES_KEY] if NFRAMES_KEY in batch_dict else None
        self.label_idx = batch_dict[KEY_WORDS_LEBEL] if KEY_WORDS_LEBEL in batch_dict else None
        self.custom_label = batch_dict[CUSTOM_LABEL] if CUSTOM_LABEL in batch_dict else None
        self.custom_label_len = batch_dict[CUSTOM_LABEL_LEN] if CUSTOM_LABEL_LEN in batch_dict else None
        self.road_n = batch_dict[ROAD_NOISE_KEY] if ROAD_NOISE_KEY in batch_dict else None
        self.p_n = batch_dict[P_NOISE_KEY] if P_NOISE_KEY in batch_dict else None
        self.s_rir = batch_dict[S_RIR_KEY] if S_RIR_KEY in batch_dict else None
        self.p_rir = batch_dict[P_NOISE_RIR_KEY] if P_NOISE_RIR_KEY in batch_dict else None


if __name__ == '__main__':
    from settings.config import *
    dataset = CZDataset(PIN_YIN_CONFIG_PATH, TRAINING_KEY_WORDS, TRAINING_BACKGROUND, TRAINING_NOISE, TRAINING_RIR, POINT_NOISE_PATH,
                        sample_rate=16000, speech_seconds=10)
    batch_dataloader = BatchDataLoader(dataset, batch_size=4, workers_num=0)
    car_zone_model_path = '/home/yanyongjie/code/official/car/car_zone_wuling_no_back_r4/model/model-302000--20.740382461547853.pickle'
    data_factory = GPUDataSimulate(TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_SNR_LIST, device='cpu', zone_model_path=car_zone_model_path)
    for batch_info in batch_dataloader:
        res = data_factory(batch_info)
        pass
