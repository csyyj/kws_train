import logging as log
import random
import torchaudio
import torch.distributed as dist
import torch.multiprocessing as mp
import scipy.signal as signal
import scipy.io as sio
import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from settings.config import *
from train.model_handle import *
from tools.batch_rir_conv import batch_rir_conv
from torch.multiprocessing import Process
from network.mdtc import MDTCSML
from tensorboardX import SummaryWriter
from simulate_data.gen_simulate_data_car_zone import BatchDataLoader, CZDataset, GPUDataSimulate
from datetime import datetime
from accelerate import Accelerator

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def gen_data_and_network(is_need_dataloader=True, model_name=None):
    set_seed(int(datetime.now().timestamp()))
    accelerator = Accelerator()
    device = accelerator.device
    net_work = MDTCSML(stack_num=4, stack_size=4, in_channels=64, res_channels=128, kernel_size=7, causal=True).to(device)
    car_zone_model_path = '/home/yanyongjie/code/official/car/car_zone_2_for_aodi_real/model/student_model/model-1200000--17.81806887626648.pickle'
    data_factory = GPUDataSimulate(TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_SNR_LIST, device=device, zone_model_path=car_zone_model_path).to(device)
    if is_need_dataloader:        
        dataset = CZDataset(TRAINING_KEY_WORDS, TRAINING_BACKGROUND, TRAINING_NOISE, TRAINING_RIR, POINT_NOISE_PATH,
                        snr_list=[-5, 10], sample_rate=16000, speech_seconds=8)
        batch_dataloader = BatchDataLoader(dataset, batch_size=BATCH_SIZE, workers_num=8)
        net_work, batch_dataloader, data_factory = accelerator.prepare(net_work, batch_dataloader, data_factory)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net_work.parameters()), lr=LR)
    optim = accelerator.prepare(optim)
    step = 0
    if RESUME_MODEL:
        step, optim_dict = resume_model(net_work, MODEL_DIR, MODEL_NAME if model_name is None else model_name, device=device)
        optim.load_state_dict(optim_dict)
    
    if is_need_dataloader:
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0
        if rank == 0 and is_need_dataloader:
            writer = SummaryWriter('runs/train_log')
        train_hidden = None
        sum_loss = 0.0
        sum_acc = 0.0
        train_hidden = None
        for batch_info in batch_dataloader:
            enhance_data, s, label_idx = data_factory(batch_info=batch_info)
            enhance_data = enhance_data.to(device)
            s = s.to(device)
            label_idx = label_idx.to(device)     
            logist, train_hidden, loss, acc = \
                net_work(enhance_data, target=label_idx, clean_speech=s, hidden=train_hidden)
                # net_work(mix=mix_in, anchor=anchor, tgt=s_in, spk_tgt=spk_tgt_tmp, spk_id=spk_id, is_spk=False, hidden=train_hidden)
            optim.zero_grad()
            accelerator.backward(loss)

            total_norm = 0
            norm_type = 2
            for k, p in net_work.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
            # print(total_norm)
            if not np.isnan(total_norm):
                torch.nn.utils.clip_grad_norm_(net_work.parameters(), 0.1)
                optim.step()
            else:
                print('grad find nan')
            sum_loss += loss.item()
            sum_acc += acc

            if step % PRINT_TIMES == 0 and rank == 0:
                avg_loss = sum_loss / PRINT_TIMES
                avg_acc = sum_acc / PRINT_TIMES
                log.info('step = {}, loss - {}, acc - {}'.format(step, avg_loss, avg_acc))
                sum_loss = 0.0
                sum_acc = 0.0
                # writer.add_scalar('loss/loss', avg_loss, step)

                # for name, param in net_work.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

            step += 1
            # to save model
            if step % TEST_TIMES == 0 and rank == 0:
                save_model(net_work.module, optim, step, avg_loss, models_dir=MODEL_DIR)
                with torch.no_grad():
                    in_np = enhance_data.to(torch.float32).detach().cpu().numpy()
                    s_np = s.to(torch.float32).detach().cpu().numpy()
                    target_np = label_idx.to(torch.long).detach().cpu().numpy()
                    data_np = np.stack([in_np, s_np], axis=-1)
                    if not os.path.exists(TRAINING_CHECK_PATH):
                        os.mkdir(TRAINING_CHECK_PATH)
                    for i in range(data_np.shape[0]):
                        sf.write('{}/{}_enhance_{}.wav'.format(TRAINING_CHECK_PATH, i, target_np[i]), data_np[i], 16000)

                
    return net_work

if __name__ == '__main__':
    # main()
    # torch.multiprocessing.set_start_method('spawn')
    # world_size= 1
    # processes = []
    # # 创建进程组
    # for rank in range(world_size):
    #     p = Process(target=gen_data_and_network, args=(rank, world_size))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    gen_data_and_network()
