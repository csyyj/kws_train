import os
import gpuRIR
import random
import string
import scipy.io as sio
import numpy as np
from multiprocessing import Pool

max_dis = 0.2
max_rt = 0.8
mic_num = 1

def gen_rir(nums):
    def gen_target_pos(max_dis, L, rc):
        while True:
            # target
            tgt_dis = random.uniform(0.05, max_dis)            
            theta = random.uniform(0 / 180 * np.pi, 180 / 180 * np.pi)
            s_y_d = np.sin(theta) * tgt_dis
            s_x_d = np.cos(theta) * tgt_dis
            s = [rc[0] + s_x_d, rc[1] + s_y_d, rc[2]]
            if s[0] > L[0] or (s[0] < 0) or (s[1] > L[1]) or (s[1] < 0) or (s[2] > L[2]) or (s[2] < 0):
                continue
            else:
                break
        return s
            
    def gen_rdm_rir():
        while True:
            c = 340
            fs = 16000
            s_max_dis = 1.5
            i_max_dis = 3
                        
            L = [random.uniform(3, 10), random.uniform(3, 10), random.uniform(2.5, 5)]
            rc = [random.uniform(1, L[0] - 1), random.uniform(1, L[1] - 1), random.uniform(1, L[2] - 1)]
            
            s1 = gen_target_pos(s_max_dis, L, rc)
            s2 = gen_target_pos(i_max_dis, L, rc)
            s3 = gen_target_pos(i_max_dis, L, rc)

            while True:
                n_p = [random.uniform(0, L[0] - 0.1), random.uniform(0, L[1] - 0.1), random.uniform(0, L[2] - 0.1)]
                if np.sqrt(np.sum((np.array(rc) - np.array(n_p)) ** 2)) > 0.5:
                    break
            
            rt_rate = random.random()
            if rt_rate < 0.5:
                rt = random.uniform(0.05, 0.2)
            elif rt_rate < 0.7:
                rt = random.uniform(0.2, 0.5)
            else:
                rt = random.uniform(0.5, max_rt)
            
            receive = np.array([rc])
            source = np.array([n_p, s1, s2, s3])
            att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
            att_max = 60.0 # Attenuation at the end of the simulation [dB]
            abs_weights = [0.9] * 5+[0.5] # Absortion coefficient ratios of the walls
            beta = gpuRIR.beta_SabineEstimation(L, rt, abs_weights=abs_weights) # Reflection coefficients
            Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, rt) # Time to start the diffuse reverberation model [s]
            Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt)	 # Time to stop the simulation [s]
            nb_img = gpuRIR.t2n(Tdiff, L)	# Number of image sources in each dimension
            h = gpuRIR.simulateRIR(L, beta, source, receive, nb_img, Tmax, fs, Tdiff=Tdiff, mic_pattern="omni")
            if np.isnan(np.sum(h)):
                continue
            dic = {'rir': h.astype(np.float32), 'L': L, 'r': receive, 'n_p': n_p, 'rt60': rt}
            break
        return dic, h.astype(np.float32)

    for i in range(nums):
        dic, h = gen_rdm_rir()
        random_suffix = ''.join(random.sample(string.ascii_letters + string.digits, 10))
        save_path = '/home/yanyongjie/train_data/rir/speaker_extraction/single'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save('{}/{}'.format(save_path, random_suffix), h)
        if i % (nums // 100) == 0:
            print('{}%'.format(i * 100 / nums))




if __name__ == '__main__':
    gen_rir(100000)
