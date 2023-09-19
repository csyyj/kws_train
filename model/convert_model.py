# -*- coding:utf-8 -*-
from ctypes import *
import sys
import numpy
import struct
import math
import numpy as np
import scipy.io as sio

# from io_utils import *

#input_file_name = './model/model-554000--10.124487104415893_export.mat'
TREAT_DCNN_AS_CNN = True


def process_lstm_weight(input):
    hidden_size = input.shape[0] // 4
    i = input[0: hidden_size]
    f = input[hidden_size:hidden_size * 2]
    g = input[hidden_size * 2:hidden_size * 3]
    o = input[hidden_size * 3:hidden_size * 4]

    # return input
    return np.concatenate((i, g, f, o), axis=0)


def process_lstm_bias(input):
    hidden_size = input.shape[0] // 4
    i = input[0: hidden_size]
    f = input[hidden_size:hidden_size * 2]
    g = input[hidden_size * 2:hidden_size * 3]
    o = input[hidden_size * 3:hidden_size * 4]

    # return input
    return np.concatenate((i, g, f, o), axis=0)


def process_conv_t(conv_t):
    a = conv_t[:, 0]
    b = conv_t[:, 1]
    c = conv_t[:, 2]
    res = np.stack([c, b, a], axis=1)
    return res

def process_bn(mdl, cnn_weight_key):
    cnn_bias_key = cnn_weight_key.replace('weight', 'bias')
    bn_weight_key = key.replace('cnn', 'bn')
    bn_bias_key = cnn_bias_key.replace('cnn', 'bn')
    bn_mean_key = bn_bias_key.replace('bias', 'running_mean')
    bn_var_key = bn_bias_key.replace('bias', 'running_var')
    cnn_weight = mdl[cnn_weight_key]
    cnn_bias = np.reshape(mdl[cnn_bias_key],[-1, 1, 1, 1])
    bn_weight =  np.reshape(mdl[bn_weight_key], [-1, 1, 1, 1])
    bn_bias =  np.reshape(mdl[bn_bias_key], [-1, 1, 1, 1])
    bn_mean =  np.reshape(mdl[bn_mean_key], [-1, 1, 1, 1])
    bn_var = np.reshape(mdl[bn_var_key], [-1, 1, 1, 1])
    cnn_weight_new = cnn_weight / (np.sqrt(bn_var + 1e-5)) * bn_weight
    cnn_bias_new = cnn_bias * bn_weight / (np.sqrt(bn_var + 1e-5)) + bn_bias - bn_weight * bn_mean / (np.sqrt(bn_var + 1e-5))
    mdl[cnn_weight_key] = cnn_weight_new
    mdl[cnn_bias_key] = np.reshape(cnn_bias_new, [1, -1])


if __name__ == '__main__':
    input_file_name = sys.argv[1]
    mdl = sio.loadmat(input_file_name)
    key_list = []
    model_dict = {}
    for key in mdl.keys():
        value = mdl[key]
        if 'weight' in key and ('conv' in key) and ('cnn' in key):  # cnn/dcnn weight
            if len(value.shape) == 3:
                value = np.expand_dims(value, -1)
                mdl[key] = value
            bn_weight_key = key.replace('cnn', 'bn')
            if bn_weight_key in mdl:
                process_bn(mdl, key)

    for key in mdl.keys():
        key_list.append(key)
        value = mdl[key]
        if 'weight' in key and ('conv' in key) and ('cnn' in key):  # cnn/dcnn weight
            if '_t_' in key and not TREAT_DCNN_AS_CNN:  # dcnn weight
                value = np.transpose(value, (2, 3, 0, 1))
                value = process_conv_t(value)
            else:  # cnn weight
                if len(value.shape) == 4:
                    value = np.transpose(value, (2, 3, 1, 0))
                else: # TCN(1D)
                    value = np.transpose(np.expand_dims(value, -1), (2, 3, 1, 0))
        elif 'lstm_weight' in key:
            value = process_lstm_weight(value)
        elif 'lstm_bias_ih' in key:
            v1 = value
            v2 = mdl[key.replace('ih', 'hh')]
            value = v1 + v2
            value = process_lstm_bias(value.T)
            key = key.replace('ih_', '')
        if 'module' in key:
            key = key.replace('module', 'crnn')
        if 'stft' not in key and 'speaker_net' not in key and 'classifier_linear_weigh' not in key and isinstance(value, np.ndarray) and 'bn' not in key:
            model_dict[key] = value
            print('key:{}, shape:{}, min:{}, max:{}'.format(key, value.shape, np.min(value), np.max(value)))

    sio.savemat(input_file_name.replace('.mat', '_convert.mat'), model_dict)

    print('finish...')
