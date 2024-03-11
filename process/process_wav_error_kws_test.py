import numpy as np
import torch
import sys
from train.train_script_1mic import *
from tools.stft_istft import *

MIX_WAV_FILE_PATH = './process/wuling'
MIX_WAV_FILE_PATH = '/home/yanyongjie/code/official/kws/wuling/nhxl/process/error_kws_0117'
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
        split_path = os.path.split(f_path)
        cur_path = '/home/yanyongjie/code/official/kws/wuling/nhxl/process/test_error_kws'
        cur_file_name = split_path[1]
        if len(mix.shape) == 1:
            mix = np.reshape(mix, [-1, 1])
        est_kws_l = []
        count_l = []
        for c in range(mix.shape[1]):
            mix_c = torch.from_numpy(mix[:, c].astype(np.float32))
            with torch.no_grad():
                mix_c = mix_c.unsqueeze(dim=0).to('cuda:0')
                t = mix_c.size(-1)
                if t > 16000 * 200:
                    l = []
                    ll = []
                    hidden = None
                    for i in range(t // (16000 * 200) + 1):
                        tmp_in = mix_c[..., i * 16000 * 200:(i + 1) * 16000 * 200]
                        if tmp_in.size(0) < 1:
                            break
                        logist, pinyin_logist, hidden, _, _, _, _ = net_work(tmp_in, hidden=hidden)
                        l.append(logist)
                        ll.append(pinyin_logist)
                    est = torch.cat(l, dim=1)
                    est_pinyin = torch.cat(ll, dim=1)
                else:
                    est, est_pinyin, _, _, _, _, _ = net_work(mix_c)
                est_logist, max_idx = (torch.softmax(est, dim=-1).squeeze()[:, 1:2]).max(1)
                est_pinyin_logist = torch.softmax(est_pinyin, dim=-1).squeeze()
                est_kws = torch.zeros_like(mix_c)
                count = 0
                k = 0
                while k < est_logist.size(0):
                    if est_logist[k] > THRES_HOLD:# and est_pinyin_logist[k - 10:k+10, 179].amax() > 0.3:
                        est_kws[:, k * 256] = 0.1 * (max_idx[k] + 1)                        
                        count += 1
                        print(est_logist[k:k+10])
                        error_kws_wav = mix_c[:, max((k - 560) * 256, 0):min((k + 25)* 256, mix_c.size(1))].squeeze().detach().cpu().numpy()
                        save_path = os.path.join(cur_path, '{}_{}_{}_{}.wav'.format(cur_file_name.replace('.wav', ''), c, count, k))
                        #sf.write(save_path, error_kws_wav, 16000)
                        k += 20
                    else:
                        k += 1
                count_l.append(count)
            min_len = min(est_kws.shape[1], mix_c.reshape(-1).shape[0])
            est_kws_l.append(mix_c.reshape(-1)[:min_len])
            est_kws_l.append(est_kws.reshape(-1)[:min_len])
        est = torch.stack(est_kws_l, dim= -1).detach().cpu().numpy()
        print(count_l)
        #sf.write(f_path.replace('.wav', '{}.wav'.format(PROCESS_EXT)), est, f)
        print('{} has process!'.format(f_path))