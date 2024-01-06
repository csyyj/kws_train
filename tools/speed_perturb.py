import torch
import torchaudio
import numpy as np
import soundfile as sf

WAV_PATH = '/home/yanyongjie/code/official/kws/xiaojingtongxue/process/sow/SOW_Hspeed/AEC_OUT_SOW_Hspeed_AEC_XWXW_100_vsp_elevoc.wav'

if __name__ == '__main__':
    data, fs = sf.read(WAV_PATH)
    data_tensor = torch.from_numpy(data.astype(np.float32).T)
    data_tensor = torchaudio.sox_effects.apply_effects_tensor(
        data_tensor, fs, [['speed', str(0.9)], ['rate', str(16000)]]
    )
    print(data.shape)
    print(data_tensor.shape)