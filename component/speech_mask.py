import torch
import numpy as np
import soundfile as sf
from component.time_vad import TimeVad

class SpeechMask(object):
    def __init__(self, stft, threshold=7, shift=80):
        super(SpeechMask, self).__init__()
        self.threshold = threshold
        self.time_vad = TimeVad(shift)
        self.stft = stft
    
    def __call__(self, wav, mask=None):
        label_vad, _ = self.time_vad(wav)
        spec = self.stft.transform(wav)
        s_pow = (spec ** 2).sum(-1)
        if mask is None:
            mask = torch.ones_like(s_pow)
        pow_db = 10 * torch.log10(s_pow + 1e-7)
        sorted, _ = torch.sort(pow_db, dim=2)
        max_pow = sorted[:, :, -pow_db.size(1) // 40:].mean(2, keepdim=True) # max top 200 Hz
        min_pow = sorted[:, :, :pow_db.size(1) // 40].mean(2, keepdim=True) # min top 200 Hz
        threshold = torch.maximum(max_pow - self.threshold, min_pow + 20)
        # zero_check = torch.where(mean_pow < -69, torch.zeros_like(mean_pow), torch.ones_like(mean_pow))
        s_mask = torch.where(pow_db > threshold, torch.ones_like(mask), torch.zeros_like(mask)) * label_vad[:, :mask.size(1)]
        fix_mask = s_mask * mask
        return fix_mask


if __name__ == '__main__':
    from tools.stft_istft2 import STFT
    stft = STFT(320, 160)
    speech_mask = SpeechMask(stft, shift=80)
    path = '/home/imu_zhangtailong/yyj/code/official/self_excited_cancel/mask_train/check/1_s.wav'
    data, fs = sf.read(path)
    in_data = torch.from_numpy(data.astype(np.float32)).unsqueeze(dim=0)
    spec = stft.transform(in_data)
    real_mask = speech_mask(in_data)
    spec_mask = spec * real_mask.unsqueeze(dim=-1)
    new_data = stft.inverse(spec_mask).squeeze().detach().cpu().numpy()
    all_data = np.stack([data, new_data[:data.shape[0]]], axis=-1)
    sf.write('/home/imu_zhangtailong/yyj/code/official/self_excited_cancel/mask_train/check.wav', all_data, fs)
    