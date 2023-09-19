import numpy as np
import torch

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

if __name__ == '__main__':
    stft = STFT(512, 256)
    import soundfile as sf
    # data, fs = sf.read('./check/0_s.wav')
    data = np.random.randn(16000*4)
    in_data = torch.from_numpy(data.astype(np.float32)).unsqueeze(dim=0)
    out_data = stft(in_data)
    out_wav = out_data.squeeze().detach().cpu().numpy()
    sf.write('out.wav', out_wav, 16000)