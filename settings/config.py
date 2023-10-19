import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIN_YIN_CONFIG_PATH = '/home/yanyongjie/code/official/kws/kws_custom/pin_yin_config.txt'

TRAINING_BACKGROUND = [
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/pinyin_all.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/wenetspeech_pinyin.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/爱宝爱宝.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/你好悠悠.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/魔方魔方.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/小创小创.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/小艺小艺.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/小薇小薇.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/小贝小贝.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/支付宝扫一扫.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/支付宝支付.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/微信支付.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/上一个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/上一曲.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/上一首.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/下一个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/下一曲.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/下一首.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/二D模式.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/三D模式.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/停止导航.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/停止播放.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/关闭声音.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/关闭空调.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/关闭透传.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/关闭降噪.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/减小音量.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/取消.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/取消全览.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/取消导航.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/向右.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/向左.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/回到桌面.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/增大音量.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/大点声.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/导航去公司.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/导航回家.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/小憩模式.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/小点声.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/开启透传.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/开启降噪.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/开始导航.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/开始播放.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/开始泊车.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/快速净化.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/我要听歌.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/打开地图.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/打开声音.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/打开导航.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/打开空调.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/挂断.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/挂断电话.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/接听.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/接听电话.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/播放下一频道.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/播放下一首.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/播放音乐.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/收藏歌曲.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/收藏这首歌.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/放大地图.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/暂停播放.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/查看全程.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/查看全览.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/正北朝上.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/确定.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第一个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第二个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第三个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第四个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第五个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第六个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第七个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第八个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第九个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第十个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第十一个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/第十二个.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/继续导航.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/继续播放.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/缩小地图.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/订阅专辑.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/调大音量.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/调小音量.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/车头朝上.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/还有多久.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/还有多远.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/退出全程.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/退出全览.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/退出导航.pickle',
       ]

TRAINING_KEY_WORDS = [#'/mnt/raid2/user_space/yanyongjie/asr/pickle/理想同学.pickle',
                      ['/mnt/raid2/user_space/yanyongjie/asr/pickle/你好奥迪.pickle',
                     '/mnt/raid2/user_space/yanyongjie/asr/pickle/你好奥迪导航回家.pickle'],
                      '/mnt/raid2/user_space/yanyongjie/asr/pickle/理想同学.pickle',
                      ]

TRAINING_NOISE = '/home/yanyongjie/train_data/car_zone/noise/'
NOISE_PARTS_NUM = 20
POINT_NOISE_PATH = '/home/yanyongjie/train_data/noise'
TRAINING_RIR = {
       'L2': ['/home/yanyongjie/train_data/car_zone/rir/aodi/L2'],
       'R2': ['/home/yanyongjie/train_data/car_zone/rir/aodi/R2'],
}
TRAINING_CHECK_PATH = './check'

ROAD_SNR_LIST = [-3, -2, -1, 0, 1, 2, 3, 5, 10]
POINT_SNR_LIST = [2, 3, 5, 10]

BATCH_SIZE = 32 
LR = 1e-6

RESUME_MODEL = True 

MODEL_DIR = './model/student_model'
MODEL_NAME = ''#'model-238500-0.08611498154699802'#'model-247500--34.434267597198485'#'model-325500--12.769076719284058'#'model-319500--16.160415515899658'#'model-319500--16.160415515899658'  # 'model-26400--15.593444061279296'  # 'model-364145-0.03338756449520588'  # 'model-307000-0.1684750882536173'

PRINT_TIMES = 100

TEST_TIMES = 500

TRAIN_FRQ_RESPONSE = '/home/yanyongjie/train_data/fir/fir_1000000.npy'
