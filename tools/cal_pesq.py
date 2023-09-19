import os
from tools.pesq import pesq
import numpy as np
import soundfile as sf

WAV_PATH = '/tmp/data'


def gen_target_file_list(target_dir, target_ext='.wav', limit='elevoc_process'):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == target_ext and '._' not in f and limit in f:
                l.append(f)
    return l


if __name__ == '__main__':
    enhance_pesq_sum = 0
    mix_pesq_sum = 0
    pesq_dict = {}
    i = 0
    all_file = gen_target_file_list(WAV_PATH)
    all_file.sort()
    for (index, path) in enumerate(all_file):
        if 'elevoc_process' in path and ('_0' in path) and ('speech' not in path):
            clean_path = path.replace('elevoc_process', '').replace('_mix_', '_speech_')
            mix_path = path.replace('elevoc_process', '')
            enhance_pesq, _ = pesq(reference=clean_path, degraded=path, sample_rate=16000)
            mix_pesq, _ = pesq(reference=clean_path, degraded=mix_path, sample_rate=16000)
            enhance_pesq_sum += enhance_pesq
            mix_pesq_sum += mix_pesq
            str_ = 'file:{}, ({}, {})'.format(path, mix_pesq, enhance_pesq)
            pesq_dict[path] = str_
            print(str_)
            i += 1

    print('mix :{}, enhance :{}'.format(mix_pesq_sum / i, enhance_pesq_sum / i))
