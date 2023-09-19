import os
import random
import string
import numpy as np
import soundfile as sf
import scipy.io as sio
import scipy.signal as signal

def gen_iir():
    r1, r2, r3, r4 = (random.uniform(-3 / 8, 3 / 8) for i in range(4))
    b = np.array([1.0, r1, r2])
    a = np.array([1.0, r3, r4])
    return b, a

def offline_fir(nums=100):
    s = np.zeros(256)
    s[0] = 1.0
    process = max(1, nums // 100)
    l = []
    for i in range(nums):
        b, a = gen_iir()
        fir = signal.lfilter(b, a, s)
        l.append(fir.astype(np.float32))
        # random_suffix = ''.join(random.sample(string.ascii_letters + string.digits, 10))
        # save_dir = '/home/imu_zhangtailong/yyj/code/official/self_excited_cancel/baseline/tools/simualte_response'
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # path = '{}/{}.max'.format(save_dir, random_suffix)
        # sio.savemat(path, {'b': b, 'a': a, 'fir': fir})
        if i % process == 0:
            print("{}%".format(i / process ))
    fir_all = np.stack(l, axis=-1)
    np.save('/home/imu_zhangtailong/yyj/code/official/self_excited_cancel/baseline/tools/fir_{}'.format(nums), fir_all.astype(np.float32))

def check_iir_fir():
    data, fs = sf.read('/home/imu_zhangtailong/yyj/code/official/self_excited_cancel/baseline/train/check/0_s.wav')
    data = data / np.max(np.abs(data))
    b, a = gen_iir()
    data_iir = signal.lfilter(b, a, data)

    s = np.zeros(256)
    s[0] = 1.0
    fir = signal.lfilter(b, a, s)
    data_fir = signal.fftconvolve(data.astype(np.float32), fir.astype(np.float32))

    min_len = min(min(data.shape[0], data_iir.shape[0]), data_fir.shape[0])
    data_all = np.stack([data[:min_len], data_iir[:min_len], data_fir[:min_len]], axis=1)
    sf.write('/home/imu_zhangtailong/yyj/code/official/self_excited_cancel/baseline/tools/check.wav', data_all, fs)
    diff = np.max(np.abs((data_fir[:min_len] - data_iir[:min_len])))
    # print(diff)
    return diff

if __name__ == '__main__':
    # max_diff = 9e-16
    # for i in range(1000):
    #     diff = check_iir_fir()
    #     max_diff = max(max_diff, diff)
    # print(max_diff)
    offline_fir(1000000)