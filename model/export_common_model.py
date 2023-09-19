import sys
from functools import reduce

import torch
import pickle
import numpy as np
from collections import OrderedDict
import scipy.io as scio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def save_info_to_mat(key, value):
    v = value if isinstance(value, np.ndarray) else value.numpy()
    scio.savemat('./{}.mat'.format(key), {key: v})


if __name__ == '__main__':
    model_name = sys.argv[1]
    state_dict = torch.load(model_name, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, value in state_dict['state_dict'].items():
        new_state_dict[k] = value
        print('{} - {}'.format(k, value.shape))

    var_list = []
    var_dict = {}
    var_size = 0
    for key in new_state_dict.keys():
        v = new_state_dict[key].cpu().numpy()
        if np.isnan(np.sum(v)):
            print('find nan: {}'.format(key))
        var_size += v.size
        var_list.append([key, v.shape, v])
        key = key.replace('.', '_')
        var_dict[key] = v

    with open('./' + model_name.replace('.pickle', '_export.mat'), 'wb') as handle:
        scio.savemat(handle, var_dict)
    print('finish, model size: {}k'.format(var_size * 4 // 1024))
