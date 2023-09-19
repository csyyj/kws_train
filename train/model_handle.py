import os
import torch
import logging as log
from collections import OrderedDict
from collections import *


def save_model(net, optim, step, loss, models_dir):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save({
        'step': step,
        'state_dict': net.state_dict(),
        'optimizer': optim.state_dict()},
        '{}/model-{}-{}.pickle'.format(models_dir, step, loss))
    log.info('save model-{}-{} success'.format(step, loss))


def resume_model(net, models_dir, resume_model_name, device='cpu'):
    log.info('resuming model...')
    models = {}
    for f in os.listdir(models_dir):
        if os.path.splitext(f)[1] == '.pickle':
            f = '{}/{}'.format(models_dir, f)
            id = os.path.getctime(f)
            models[id] = f
    if len(models) < 1:
        log.info('there is no models to resume')
        return
    if len(resume_model_name) > 0:
        model_name = '{}/{}.pickle'.format(models_dir, resume_model_name)
    else:
        index = sorted(models)[-1]
        model_name = models[index]
    # if is_map_to_cpu or not torch.cuda.is_available():
    model_dict = torch.load(model_name, map_location=device)
    state_dict = model_dict['state_dict']
    for k, v in net.state_dict().items():
        if k.split('.')[0] == 'module':
            net_has_module = True
        else:
            net_has_module = False
        break
    dest_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split('.')[0] == 'module':
            ckpt_has_module = True
        else:
            ckpt_has_module = False
        if net_has_module == ckpt_has_module:
            dest_state_dict = state_dict
            break
        if ckpt_has_module:
            dest_state_dict[k.replace('module.', '')] = v
        else:
            dest_state_dict['module.{}'.format(k)] = v

    net.load_state_dict(dest_state_dict, False)
    step = model_dict['step']
    optim_state = model_dict['optimizer']
    log.info('finish to resume model {}.'.format(model_name))
    return step, optim_state


def mapping_model(dest_net, model_path):
    log.info('resuming model...')
    model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = model_dict['state_dict']
    dest_state_dict = OrderedDict()
    for k, v in zip(list(dest_net.state_dict().keys()), list(dest_net.state_dict().values())):
        dest_state_dict[k] = v.detach().cpu()
    new_state_dict = OrderedDict()
    print('=====================please check keys are matched===========================')
    for key1, key2 in zip(list(state_dict.keys()), list(dest_state_dict.keys())):
        print('({}, {})'.format(key1, key2))
        v1 = state_dict[key1]
        v2 = dest_state_dict[key2]
        if v1.shape != v2.shape:
            if v1.numpy().size == v2.numpy().size:
                v1 = v1.reshape(v2.shape)
                print('{} has reshape from {} to {}'.format(key2, v2.shape, v1.shape))
            elif len(v1.shape) == 2 and (v1.numpy().shape[0] % 8 != 0 or v1.numpy().shape[1] % 8 != 0):
                if v1.shape[0] != v2.shape[0]:
                    v1 = torch.cat([v1, torch.zeros([v2.shape[0] - v1.shape[0], v1.shape[1]])], dim=0)
                if v1.shape[1] != v2.shape[1]:
                    v1 = torch.cat([v1, torch.zeros([v1.shape[0], v2.shape[1] - v1.shape[1]])], dim=1)
                assert v1.shape == v2.shape, 'paddding error: ({}, {})'.format(key1, key2)
            elif len(v1.shape) == 1 and v1.numpy().shape[0] % 8 != 0:
                v1 = torch.cat([v1, torch.zeros(v2.shape[0] - v1.shape[0])], dim=0)
            else:
                log.error("process error! ({}, {})".format(key1, key2))
        new_state_dict[key2] = v1
    print('=============================================================================')
    dest_net.load_state_dict(new_state_dict, False)
    log.info('finish to resume model.')
    return True
