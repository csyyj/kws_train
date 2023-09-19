import os
import shutil
import random
import numpy as np
import soundfile as sf
import scipy.signal as signal
from multiprocessing import Pool

WAV_LIST = '/home/imu_zhangtailong/ddn/data_yj/speech/LibriSpeech'


def gen_speaker_list(target_dir, target_ext='.flac'):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == target_ext and '._' not in f:
                if root not in l:
                    l.append(root)
    return l


def gen_wav_list(target_dir, target_ext='.flac'):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == target_ext and '._' not in f:
                if root not in l:
                    l.append(f)
    return l


def rm_error_file(target_dir):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == '.npy' and '._' not in f and 'cat_speaker' in f:
                # os.remove(f)
                l.append(f)
    return l


def process_cat(speaker_list):
    for i, path in enumerate(speaker_list):
        file_name = os.path.split(path)[1]
        dest_file_path = os.path.join(path, 'cat_speaker_{}.npy'.format(file_name))
        data_list = []
        for f in gen_wav_list(path):
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == '.flac' and '._' not in f:
                f_path = os.path.join(path, f)
                data, fs = sf.read(f_path)
                if fs != 16000:
                    if fs < 16000:
                        print('sample rate error')
                    else:
                        data = signal.resample(data, data.shape[0] * 16000 // fs, axis=0).astype(np.float32)

                data_list.append(data / np.max(np.abs(data) + 1e-6))

        cat_data = np.concatenate(data_list, axis=0).astype(np.float32)
        with open(dest_file_path, 'wb') as f:
            np.save(f, cat_data)
        if i % 10 == 0:
            print('{}%'.format(int(i / len(speaker_list) * 100)))



if __name__ == '__main__':
    process_list = gen_speaker_list(WAV_LIST)
    print(process_list)
    process_num = 35
    file_num_per_process = len(process_list) // process_num
    param_list = []
    for i in range(process_num):
        if i == process_num - 1:
            param_list.append(process_list[i * file_num_per_process:])
        else:
            param_list.append(process_list[i * file_num_per_process: (i + 1) * file_num_per_process])
    pool = Pool(processes=process_num)
    for x in range(process_num):
        pool.apply_async(process_cat, args=(param_list[x],))
    pool.close()
    pool.join()

