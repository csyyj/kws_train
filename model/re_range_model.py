import os
import torch
import logging as log
from collections import OrderedDict
from collections import *


def convert_model(ori_path, dest_path):
    model_dict = torch.load(ori_path, map_location='cpu')
    state_dict = model_dict['state_dict']
    dest_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'class_out' in k:
            v = torch.cat([v[0:1], v[2:3], v[1:2], v[3:]], dim=0)
        dest_state_dict[k] = v
    torch.save({
        'step': model_dict['step'],
        'state_dict': dest_state_dict,
        'optimizer': model_dict['optimizer']},
        dest_path)

if __name__ == '__main__':
    convert_model('/home/yanyongjie/code/official/kws/wuling/nhxl/model/student_model/model-2542500--21.871500091552733.pickle',
                  '/home/yanyongjie/code/official/kws/wuling/nhxl/convert.pickle')
