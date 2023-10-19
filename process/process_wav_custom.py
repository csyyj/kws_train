import numpy as np
import torch
import sys
from train.train_script_1mic import *
from tools.stft_istft import *

MIX_WAV_FILE_PATH = './process/custom'
PROCESS_EXT = 'elevoc_process'
THRES_HOLD = 0.5
eval_keywords = [['li'],
                 ['xiang'],
                 ['tong'],
                 ['xue']]

# eval_keywords = [['ni'],
#                  ['hao'],
#                  ['xiao'],
#                  ['di']]

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

def parse_pin_yin_config(pin_yin_config_path):
        with open(pin_yin_config_path, 'r') as f:
            data = f.readline()
            dict = {}
            while data:
                key, value = data.split('\t')
                dict[key] = value.replace('\n', '')
                data = f.readline()
        return dict


if __name__ == '__main__':
    decode_window_len = 100
    out_prob_len = 410
    search_interval = 100
    spot_threshold = 0.1
    decision_threshold = 0.1
    pinyin_dict = parse_pin_yin_config('/home/yanyongjie/code/official/kws/kws_custom/pin_yin_config.txt')
    
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
        count_l = []
        for i in range(mix.shape[1]):
            mix_c = torch.from_numpy(mix[:, i].astype(np.float32))
            mix_c = mix_c.unsqueeze(dim=0).to('cuda:0')
            est_kws = torch.zeros_like(mix_c)
            with torch.no_grad():
                t = mix_c.size(-1)
                if t > 16000 * 200:
                    l = []
                    hidden = None
                    for i in range(t // (16000 * 200) + 1):
                        tmp_in = mix_c[..., i * 16000 * 200:(i + 1) * 16000 * 200]
                        if tmp_in.size(0) < 1:
                            break
                        _, logist, hidden, _, _, _ = net_work(tmp_in, hidden=hidden)
                        l.append(logist)
                    est = torch.cat(l, dim=1)
                else:
                    _, est, _, _, _, _ = net_work(mix_c)
                est = torch.softmax(est, dim=-1)
            
            out_prob = est[0].cpu().detach().squeeze()
            out_prob = out_prob.numpy()
            slide_window = np.zeros([decode_window_len, out_prob_len], dtype=np.float32)
            wake_count = 0
            for i in range(out_prob.shape[0]):
                slide_window = np.concatenate([slide_window, out_prob[i: i + 1]], axis=0)
                slide_window = slide_window[1:]

                last_word = eval_keywords[-1]
                last_word_prob = np.array([slide_window[-1][int(pinyin_dict[word])] for word in last_word]).sum()

                if last_word_prob > spot_threshold:
                    prob_list = [last_word_prob]
                    idx_list = [decode_window_len - 1]

                    start_idx = (decode_window_len - 1) - 2
                    for search_keyword in eval_keywords[:-1][::-1]:
                        max_idx = -1
                        max_prob = 0
                        for idx in range(start_idx, max(start_idx - search_interval, 0), -1):
                            now_word_prob = np.array(
                                [slide_window[idx][int(pinyin_dict[word])] for word in search_keyword]).sum()
                            if now_word_prob > max_prob:
                                max_idx = idx
                                max_prob = now_word_prob
                                if max_prob > spot_threshold:
                                    break
                        if max_idx == -1 or max_prob < spot_threshold:
                            break
                        prob_list.append(max_prob)
                        idx_list.append(max_idx)
                        start_idx = max_idx - 2
                    if len(prob_list) == len(eval_keywords):
                        finial_prob = 1
                        for prob in prob_list:
                            finial_prob *= prob
                        finial_prob = finial_prob ** (1 / len(prob_list))
                        if finial_prob > decision_threshold:
                            slide_window = np.zeros_like(slide_window, dtype=np.float32)
                            hh = i // (360000)
                            mm = (i - hh * 360000) // 6000
                            ss = (i - hh * 360000 - mm * 6000) // 100
                            SS = i - hh * 360000 - mm * 6000 - ss * 100
                            # print('%02d:%02d:%02d:%02d %.6f ' % (hh, mm, ss, SS, finial_prob))
                            est_kws[:, i * 256] = 0.8
                            wake_count += 1
            est_kws_l.append(mix_c)
            est_kws_l.append(est_kws)
            count_l.append(wake_count)
        est = torch.stack(est_kws_l, dim= -1).detach().squeeze().cpu().numpy()
        print(count_l)
        sf.write(f_path.replace('.wav', '{}.wav'.format(PROCESS_EXT)), est, f)
        print('{} has process!'.format(f_path))
