import torch
import numpy as np
import soundfile as sf


class TimeVad(object):
    def __init__(self, shift=256):
        super(TimeVad, self).__init__()
        self.shift = shift

    def __frame__(self, in_wav):
        b, t = in_wav.size()
        padding_size = int(np.ceil(in_wav.shape[-1] / self.shift)) * self.shift - in_wav.shape[-1]
        if padding_size > 0:
            pad_wav = torch.cat([in_wav, torch.zeros([b, padding_size], device=in_wav.device)], dim=-1)
        else:
            pad_wav = torch.ones_like(in_wav) * in_wav
        frame_wav = pad_wav.reshape([b, -1, self.shift])
        return frame_wav

    def __call__(self, in_wav):
        frame_wav = self.__frame__(in_wav)
        pow = (frame_wav ** 2).sum(-1)
        pow_db = 10 * torch.log10(pow + 1e-7)
        sorted, indices = torch.sort(pow_db, dim=-1)
        mean_pow = sorted[:, -25:].mean(-1, keepdim=True)
        frame_vad = torch.where(pow_db <= (mean_pow - 25), torch.zeros_like(pow_db),
                                torch.ones_like(pow_db)).unsqueeze(dim=-1)
        # in_wav为全0时，mean_pow为-70
        zero_check = torch.where(mean_pow < -69, torch.zeros_like(mean_pow), torch.ones_like(mean_pow))
        frame_vad = frame_vad * zero_check.unsqueeze(dim=-1)
        sample_vad = frame_vad.repeat(1, 1, self.shift).reshape(in_wav.shape[0], -1)[:, :in_wav.shape[-1]]
        return frame_vad, sample_vad


if __name__ == '__main__':
    wav_path = '/Users/csyyj/fsdownload/check/7_speech.wav'
    data, fs = sf.read(wav_path)
    data_torch = torch.from_numpy(data).unsqueeze(dim=0)
    time_vad = TimeVad()
    _, vad = time_vad(data_torch)
    res = np.stack([data, vad.squeeze().detach().cpu().numpy() * data], axis=-1)
    sf.write('./res.wav', res, fs)
