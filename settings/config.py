import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,7,8,9"

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_BACKGROUND = '/home/yanyongjie/train_data/speech'
TRAINING_KEY_WORDS = ['/home/yanyongjie/train_data/kws_data/理想同学',
                     #  '/home/yanyongjie/train_data/kws_data/导航去公司',
                     #  '/home/yanyongjie/train_data/kws_data/导航回家',
                     #  '/home/yanyongjie/train_data/kws_data/增大音量',
                     #  '/home/yanyongjie/train_data/kws_data/减小音量',
                     #  '/home/yanyongjie/train_data/kws_data/关闭声音',
                     #  '/home/yanyongjie/train_data/kws_data/取消导航',
                      ]

TRAINING_NOISE = '/home/yanyongjie/train_data/car_zone/noise/'
NOISE_PARTS_NUM = 20
POINT_NOISE_PATH = '/home/yanyongjie/train_data/noise'
TRAINING_RIR = {
       'L1': ['/home/yanyongjie/train_data/car_zone/rir/byd_han/L1'],
       'R1': ['/home/yanyongjie/train_data/car_zone/rir/byd_han/R1'],
       'L2': ['/home/yanyongjie/train_data/car_zone/rir/byd_han/L2',
              '/home/yanyongjie/train_data/car_zone/rir/byd_han/C2_L'],
       'R2': ['/home/yanyongjie/train_data/car_zone/rir/byd_han/R2',
              '/home/yanyongjie/train_data/car_zone/rir/byd_han/C2_R'],
}
TRAINING_CHECK_PATH = './check'

ROAD_SNR_LIST = [-5, -2, -1, 0, 1, 2, 3, 5, 10]
POINT_SNR_LIST = [ 0, 1, 2, 3, 5, 10]
SNR_LIST = [-5, -2, -1, 0, 1, 2, 3, 5, 10]

BATCH_SIZE = 64
LR = 0.00001

RESUME_MODEL = True 

MODEL_DIR = './model/student_model'
MODEL_NAME = ''  # 'model-26400--15.593444061279296'  # 'model-364145-0.03338756449520588'  # 'model-307000-0.1684750882536173'

PRINT_TIMES = 100

TEST_TIMES = 1000

TRAIN_FRQ_RESPONSE = '/home/yanyongjie/train_data/fir/fir_1000000.npy'
