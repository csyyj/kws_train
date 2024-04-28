import os
import numpy as np
import soundfile as sf

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


def cal_end(in_wav, shift=256):
        def frame(in_wav, shift):
            padding_size = int(np.ceil(in_wav.shape[-1] / shift)) * shift - in_wav.shape[-1]
            if padding_size > 0:
                pad_wav = np.concatenate([in_wav, np.zeros([padding_size], dtype=np.float32)], axis=-1)
            else:
                pad_wav = np.copy(in_wav)
            frame_wav = pad_wav.reshape([-1, shift])
            return frame_wav
        
        frame_wav = frame(in_wav, shift)
        real_frame = frame_wav.shape[0]
        pow = (frame_wav ** 2).sum(-1)
        pow_db = 10 * np.log10(pow + 1e-7)
        BYPASS_FRAME_LEN = 5
        pow_db_real = pow_db[:real_frame - BYPASS_FRAME_LEN]
        threshold = pow_db_real[-10:].mean() + 20
        flip_pow_db_real = np.flip(pow_db_real, axis=-1)
        flip_pow_db_real_bool = flip_pow_db_real > threshold
        for i in range(real_frame - 10):
            if flip_pow_db_real_bool[i:i + 10].sum().item() > 9:
                break
        end = (real_frame - BYPASS_FRAME_LEN - i) * shift
        return end
    
if __name__ == '__main__':
    ORI_DIR = '/mnt/raid2/user_space/yanyongjie/asr/实采语音/小捷你好/'
    DEST_DIR = './oneshot'
    wav_list = gen_target_file_list(ORI_DIR)
    for i, path in enumerate(wav_list):
        data, fs = sf.read(path)
        end = cal_end(data)
        channel_2 = np.zeros_like(data)
        channel_2[end] = 1
        data = np.stack([data, channel_2], -1)
        file_name = os.path.split(path)[-1]
        dest_path = os.path.join(DEST_DIR, file_name)
        sf.write(dest_path, data, fs)
        if i > 1000:
            break