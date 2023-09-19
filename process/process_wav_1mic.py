import numpy as np
import torch
import sys
from train.train_script_1mic import *
from tools.stft_istft import *

MIX_WAV_FILE_PATH = './process/test_wav'
PROCESS_EXT = 'elevoc_process'
THRES_HOLD = 0.5

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


if __name__ == '__main__':
    args = sys.argv
    if len(args) >= 2:
        model_name = sys.argv[1]
    else:
        model_name = None
    net_work = gen_data_and_network(is_need_dataloader=False, model_name=model_name)
    net_work = net_work.to('cuda:0')
    net_work.eval()

    for f_path in gen_target_file_list(MIX_WAV_FILE_PATH):
        if '._' in f_path or PROCESS_EXT in f_path:
            continue
        mix, f = sf.read(f_path)
        if len(mix.shape) == 1:
            mix = np.reshape(mix, [-1, 1])
        est_kws_l = []
        for i in range(mix.shape[1]):
            mix_c = torch.from_numpy(mix[:, i].astype(np.float32))
            with torch.no_grad():
                mix_c = mix_c.unsqueeze(dim=0).to('cuda:0')
                t = mix_c.size(-1)
                if t > 16000 * 20:
                    l = []
                    hidden = None
                    for i in range(t // (16000 * 20) + 1):
                        tmp_in = mix_c[..., i * 16000 * 20:(i + 1) * 16000 * 20]
                        if tmp_in.size(0) < 1:
                            break
                        logist, hidden, _, _ = net_work(tmp_in, hidden=hidden)
                        l.append(logist)
                    est = torch.cat(l, dim=1)
                else:
                    est, _, _, _ = net_work(mix_c)
                est_logist = torch.softmax(est, dim=-1).squeeze()[:, 1]
                est_kws = torch.zeros_like(est_logist)
                est_kws[est_logist > THRES_HOLD] = 0.8
                est_kws = torch.tile(est_kws.reshape(-1, 1), [1, 256]).reshape(-1)
            min_len = min(est_kws.shape[0], mix_c.reshape(-1).shape[0])
            est_kws_l.append(mix_c.reshape(-1)[:min_len])
            est_kws_l.append(est_kws[:min_len])
        est = torch.stack(est_kws_l, dim= -1).detach().cpu().numpy()
        sf.write(f_path.replace('.wav', '{}.wav'.format(PROCESS_EXT)), est, f)
        print('{} has process!'.format(f_path))
