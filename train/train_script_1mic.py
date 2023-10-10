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
from network.mdtc_8bit_ctc import MDTCSML
from tensorboardX import SummaryWriter
from simulate_data.gen_simulate_data_car_zone import BatchDataLoader, CZDataset, GPUDataSimulate
from datetime import datetime
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def gen_data_and_network(is_need_dataloader=True, model_name=None):
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    device = accelerator.device
    net_work = MDTCSML(stack_num=4, stack_size=4, in_channels=64, res_channels=128, kernel_size=7, causal=True).to(device)
    car_zone_model_path = '/home/yanyongjie/code/official/car/car_zone_2_for_aodi_2_zone_real/model/student_model/model-222000--17.185455236434937.pickle'
    data_factory = GPUDataSimulate(TRAIN_FRQ_RESPONSE, ROAD_SNR_LIST, POINT_SNR_LIST, device=device, zone_model_path=car_zone_model_path).to(device)
    if is_need_dataloader:
        try:
            rank = torch.distributed.get_rank()
        except:
            rank = 0
            print('set rank 0')
        set_seed(int(datetime.now().timestamp()) + rank)        
        dataset = CZDataset(PIN_YIN_CONFIG_PATH, TRAINING_KEY_WORDS, TRAINING_BACKGROUND, TRAINING_NOISE, TRAINING_RIR, POINT_NOISE_PATH,
                        sample_rate=16000, speech_seconds=10)
        batch_dataloader = BatchDataLoader(dataset, batch_size=BATCH_SIZE, workers_num=8)
        net_work, batch_dataloader, data_factory = accelerator.prepare(net_work, batch_dataloader, data_factory)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net_work.parameters()), lr=LR)
    optim = accelerator.prepare(optim)
    step = 0
    if RESUME_MODEL:
        step, optim_dict = resume_model(net_work, MODEL_DIR, MODEL_NAME if model_name is None else model_name, device=device)
        optim.load_state_dict(optim_dict)
    
    if is_need_dataloader:
        if rank == 0 and is_need_dataloader:
            writer = SummaryWriter('runs/train_log')
        train_hidden = None
        sum_loss = 0.0
        sum_acc = 0.0
        sum_c_acc = 0.0
        train_hidden = None
        for batch_info in batch_dataloader:
            enhance_data, s, label_idx, custom_label, custom_label_len, real_frames = data_factory(batch_info=batch_info)
            enhance_data = enhance_data.to(device)
            s = s.to(device)
            label_idx = label_idx.to(device)
            custom_label = custom_label.to(device)
            custom_label_len = custom_label_len.to(device)
            real_frames = real_frames.to(device)
            # enhance_data = torch.cat([enhance_data, s], dim=0)
            # s = torch.cat([s, s], dim=0)
            # label_idx = torch.cat([label_idx, label_idx], dim=0)   
            # wav, kw_target=None, ckw_target=None, real_frames=None, ckw_len=None, clean_speech=None, hidden=None, custom_in=  
            logist, _,  train_hidden, loss, acc, acc2 = \
                net_work(enhance_data, kw_target=label_idx, ckw_target=custom_label, ckw_len=custom_label_len, real_frames=real_frames, clean_speech=s, hidden=train_hidden)
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
                torch.nn.utils.clip_grad_norm_(net_work.parameters(), 0.01)
                optim.step()
            else:
                print('grad find nan')
            sum_loss += loss.item()
            sum_acc += acc
            sum_c_acc += acc2

            if step % PRINT_TIMES == 0 and rank == 0:
                avg_loss = sum_loss / PRINT_TIMES
                avg_acc = sum_acc / PRINT_TIMES
                avg_c_acc = sum_c_acc / PRINT_TIMES
                log.info('step = {}, loss - {}, acc - {}, c_acc - {}'.format(step, avg_loss, avg_acc, avg_c_acc))
                sum_loss = 0.0
                sum_acc = 0.0
                sum_c_acc = 0.0
                # writer.add_scalar('loss/loss', avg_loss, step)

                # for name, param in net_work.named_parameters():
                #     writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

            step += 1
            # to save model
            if step % TEST_TIMES == 0 and rank == 0:
                save_model(net_work.module, optim, step, avg_loss, models_dir=MODEL_DIR)
                with torch.no_grad():
                    # mix_np = mix_data.to(torch.float32).detach().cpu().numpy()
                    in_np = enhance_data.to(torch.float32).detach().cpu().numpy()
                    s_np = s.to(torch.float32).detach().cpu().numpy()
                    target_np = label_idx.to(torch.long).detach().cpu().numpy()
                    # data_np = np.stack([in_np, s_np], axis=-1)
                    if not os.path.exists(TRAINING_CHECK_PATH):
                        os.mkdir(TRAINING_CHECK_PATH)
                    for i in range(s_np.shape[0]):
                        sf.write('{}/{}_enhance_{}.wav'.format(TRAINING_CHECK_PATH, i, target_np[i]), in_np[i], 16000)
                        # sf.write('{}/{}_mix_{}.wav'.format(TRAINING_CHECK_PATH, i, target_np[i]), mix_np[i], 16000)

                
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
