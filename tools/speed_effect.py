import torch
import torchaudio
import numpy as np
import soundfile as sf


if __name__ == '__main__':
    data, fs = sf.read('./tools/data.wav')
    data_tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(dim=0)
    sox_rate = 0.97
    data_new, _ =torchaudio.sox_effects.apply_effects_tensor(
                    data_tensor, 16000, [['speed', str(sox_rate)], ['rate', str(16000)]]
                    )
    sf.write('./tools/res.wav', data_new.squeeze().detach().cpu().numpy(), fs)