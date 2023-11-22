import os
import pickle

WAV_PATH = '/mnt/raid2/user_space/yanyongjie/asr/keywords_marked/你好奥迪'
PICKLE_PATH = '/mnt/raid2/user_space/yanyongjie/asr/keywords_marked/你好奥迪.pickle'

def gen_target_file_list(target_dir, target_ext='.wav'):
    l = []
    for root, dirs, files in os.walk(target_dir, followlinks=True):
        for f in files:
            f = os.path.join(root, f)
            ext = os.path.splitext(f)[1]
            ext = ext.lower()
            if ext == target_ext and '._' not in f:
                l.append(f)
    return l

if __name__ == '__main__':
    wav_l = gen_target_file_list(WAV_PATH)
    with open(PICKLE_PATH, 'wb') as f:
        l = []
        for idx, path in enumerate(wav_l):
            tmp = ['nihaoaodi_{}'.format(idx), path.replace('/mnt/raid2/user_space/yanyongjie/asr/', ''), 'ni hao ao di', '你好奥迪']
            l.append(tmp)
        pickle.dump(l, f)