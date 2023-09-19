import os

import shutil

TXT_PATH1 = '/home/zhangpeng/raid/dataset/car_data/lixiang/train_data/lixiangkws2/longmao_16k/total_wav.txt'
TXT_PATH2 = '/home/zhangpeng/raid/dataset/car_data/lixiang/train_data/lixiangkws2/lixiangkws2_add_20220111_16k/total_wav.txt'
TXT_PATH3 = '/home/zhangpeng/raid/dataset/car_data/lixiang/train_data/part1_16k/total_wav.txt'
DEST_PATH = '/mnt/raid2/user_space/yanyongjie/kws_data'
EXT = 'longmao'

def analyze_txt(txt_path):
    dict = {}
    with open(txt_path, 'r') as f:
        for context in f:
            key = context.split('\t')[-1].replace('\n', '')
            path = context.split('\t')[0]
            if key in dict:
                dict[key].append(path)
            else:
                dict[key] = [path]
                
    for key, paths in dict.items():
        dest_path = '{}/{}'.format(DEST_PATH, key)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for path in paths:
            name = os.path.basename(path)
            file_path = os.path.join(dest_path, name)
            shutil.copyfile(path, file_path)
        print(key)

if __name__ == '__main__':
    txt_l = [TXT_PATH1, TXT_PATH2, TXT_PATH3]
    for txt_path in txt_l:
        analyze_txt(txt_path)