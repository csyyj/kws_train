import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tools.torch_stft import STFT
# from component.time_vad import TimeVad


class TimeVad(object):
    def __init__(self, shift=256):
        super(TimeVad, self).__init__()
        self.shift = shift

    def __frame__(self, in_wav):
        b, t = in_wav.size()
        padding_size = int(np.ceil(in_wav.shape[-1] / self.shift)) * self.shift - in_wav.shape[-1]
        if padding_size > 0:
            pad_wav = torch.cat([in_wav, torch.zeros([b, padding_size], device=in_wav.device)], dim=-1)
        else:
            pad_wav = torch.ones_like(in_wav) * in_wav
        frame_wav = pad_wav.reshape([b, -1, self.shift])
        return frame_wav

    def __call__(self, in_wav):
        frame_wav = self.__frame__(in_wav)
        pow = (frame_wav ** 2).sum(-1)
        pow_db = 10 * torch.log10(pow + 1e-7)
        sorted, indices = torch.sort(pow_db, dim=-1)
        mean_pow = sorted[:, -100:].mean(-1, keepdim=True)
        frame_vad = torch.where(pow_db <= (mean_pow - 25), torch.zeros_like(pow_db),
                                torch.ones_like(pow_db)).unsqueeze(dim=-1)
        # in_wav为全0时，mean_pow为-70
        zero_check = torch.where(mean_pow < -69, torch.zeros_like(mean_pow), torch.ones_like(mean_pow))
        frame_vad = frame_vad * zero_check.unsqueeze(dim=-1)
        sample_vad = frame_vad.repeat(1, 1, self.shift).reshape(in_wav.shape[0], -1)[:, :in_wav.shape[-1]]
        return frame_vad, sample_vad

CHANNEL = 20
CHANNEL2 = 16

class TCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1, out_ac=None, is_bn=True):
        super(TCNConv, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding=0, dilation=dilation, bias=True)
        self.padding = padding
        self.ac = out_ac

    def forward(self, in_feat, hidden=None):
        if hidden is None:
            pad_feat = F.pad(in_feat, [self.padding, 0])
        else:
            pad_feat = torch.cat([hidden, in_feat], dim=-1)
            
        res = self.cnn(pad_feat)
        if self.ac is not None:
            res = self.ac(res)
        return res, pad_feat[:, :, -self.padding:]


class SpeechConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, out_ac=None, is_bn=True):
        super(SpeechConv, self).__init__()
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, bias=True)
        self.ac = out_ac

    def forward(self, in_feat):
        res = self.cnn(in_feat)
        if self.ac is not None:
            res = self.ac(res)
        return res


class SpeechDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=None, out_ac=None, is_bn=True):
        super(SpeechDeConv, self).__init__()
        self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                             padding=padding, bias=True)
        self.ac = out_ac

    def forward(self, in_feat):
        res = self.cnn(in_feat)
        if self.ac is not None:
            res = self.ac(res)
        b, c, t, f = res.size()
        res = res.reshape(b, 2, c // 2, t, f).permute(0, 2, 3, 4, 1).reshape(b, c // 2, t, -1)
        return res


class NetCRNN(nn.Module):
    def __init__(self):
        super(NetCRNN, self).__init__()

        self.layer_norm = nn.LayerNorm([5 * 2 * 2, 8]) # [5 * 2(mic) * 2 (ri), 4]

        self.conv1 = SpeechConv(in_channels=20, out_channels=CHANNEL, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1), out_ac=nn.PReLU(CHANNEL)) # 128

        self.conv2 = SpeechConv(in_channels=CHANNEL, out_channels=CHANNEL, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1), out_ac=nn.PReLU(CHANNEL)) # 64

        self.conv3 = SpeechConv(in_channels=CHANNEL, out_channels=CHANNEL, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1), out_ac=nn.PReLU(CHANNEL)) # 32

        self.conv4 = SpeechConv(in_channels=CHANNEL, out_channels=CHANNEL2, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1), out_ac=nn.PReLU(CHANNEL2)) # 16
        
        self.conv5 = SpeechConv(in_channels=CHANNEL2, out_channels=CHANNEL2, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1), out_ac=nn.PReLU(CHANNEL2)) # 8
        
        self.conv6 = SpeechConv(in_channels=CHANNEL2, out_channels=CHANNEL2, kernel_size=(1, 3), stride=(1, 2),
                                padding=(0, 1), out_ac=nn.PReLU(CHANNEL2)) # 4
        

        self.tcn_conv1 = TCNConv(CHANNEL2 * 4, CHANNEL2 * 4, kernel_size=3, padding=2, dilation=1, out_ac=nn.PReLU(CHANNEL2 * 4))
        self.tcn_conv2 = TCNConv(CHANNEL2 * 4, CHANNEL2 * 4, kernel_size=3, padding=4, dilation=2, out_ac=nn.PReLU(CHANNEL2 * 4))
        self.tcn_conv3 = TCNConv(CHANNEL2 * 4, CHANNEL2 * 4, kernel_size=3, padding=8, dilation=4, out_ac=nn.PReLU(CHANNEL2 * 4))
        self.tcn_conv4 = TCNConv(CHANNEL2 * 4, CHANNEL2 * 4, kernel_size=3, padding=16, dilation=8, out_ac=nn.PReLU(CHANNEL2 * 4))
        self.tcn_conv5 = TCNConv(CHANNEL2 * 4, CHANNEL2 * 4, kernel_size=3, padding=32, dilation=16, out_ac=nn.PReLU(CHANNEL2 * 4))
        self.tcn_conv6 = TCNConv(CHANNEL2 * 4, CHANNEL2 * 4, kernel_size=3, padding=64, dilation=32, out_ac=nn.PReLU(CHANNEL2 * 4))
        self.lstm = nn.LSTM(input_size=CHANNEL2 * 4, hidden_size=CHANNEL2 * 4, num_layers=2, batch_first=True)

        self.conv6_t = SpeechDeConv(in_channels=CHANNEL2 * 2, out_channels=CHANNEL2 * 2, kernel_size=(1, 3),
                                    padding=(0, 1), out_ac=nn.PReLU(CHANNEL2 * 2))
        
        self.conv5_t = SpeechDeConv(in_channels=CHANNEL2 * 2, out_channels=CHANNEL2 * 2, kernel_size=(1, 3),
                                    padding=(0, 1), out_ac=nn.PReLU(CHANNEL2 * 2))
        
        self.conv4_t = SpeechDeConv(in_channels=CHANNEL2 * 2, out_channels=CHANNEL * 2, kernel_size=(1, 3),
                                    padding=(0, 1), out_ac=nn.PReLU(CHANNEL * 2))

        self.conv3_t = SpeechDeConv(in_channels=CHANNEL * 2, out_channels=CHANNEL * 2, kernel_size=(1, 3),
                                    padding=(0, 1), out_ac=nn.PReLU(CHANNEL * 2))

        self.conv2_t = SpeechDeConv(in_channels=CHANNEL * 2, out_channels=CHANNEL * 2, kernel_size=(1, 3),
                                    padding=(0, 1), out_ac=nn.PReLU(CHANNEL * 2))

        self.conv1_t_mask = SpeechDeConv(in_channels=CHANNEL * 2, out_channels= 2 * 2, kernel_size=(1, 3),
                                    padding=(0, 1), out_ac=torch.sigmoid)
        
        self.conv1_t_angle = SpeechDeConv(in_channels=CHANNEL * 2, out_channels=2 * 4, kernel_size=(1, 3),
                                    padding=(0, 1))
        self.vad_fc = nn.Linear(CHANNEL2 * 4, 2)
        
        self.stft = STFT(512, 256)
        self.stft_for_loss = STFT(512, 256)

    def forward(self, mix_wav, label_speech=None, hidden=None):
        '''
        :param mix_wav: [B, 2, T]
        :param label_speech:
        :param hidden:
        :return:
        '''
        if hidden is None:
            hidden = [None for i in range(7)]
        else:
            for i in range(mix_wav.size(0)):
                if random.random() < 0.2:
                    for j in range(len(hidden)):
                        if isinstance(hidden[j], torch.Tensor):
                            hidden[j][i] = torch.zeros_like(hidden[j][i])
                        elif isinstance(hidden[j], tuple):
                            for k in range(len(hidden[j])):
                                hidden[j][k][:, i] = torch.zeros_like(hidden[j][k][:, i])


        with torch.no_grad():
            b, c, t = mix_wav.shape

            mix_reshape = mix_wav.reshape(b * c, t)
            mix_spec = self.stft.transform(mix_reshape)
            _, t, f, _ = mix_spec.shape
            mix_spec = mix_spec.reshape(b, c, t, f, -1)
            in_spec = mix_spec.permute(0, 1, 4, 2, 3).reshape(b, -1, t, f)
            
            pad_feat = F.pad(in_spec[:, :, :, 1:], [0, 0, 0, 4])
            in_feat = torch.stack([pad_feat[:, :, :-4], pad_feat[:, :, 1:-3], pad_feat[:, :, 2:-2], pad_feat[:, :, 3:-1], pad_feat[:, :, 4:]], dim=1).reshape(b, 5 * 2 * 2, t, -1)

        norm_feat = self.layer_norm(in_feat.reshape(b, 20, t, 32, 8).permute(0, 2, 3, 1, 4)).permute(0, 3, 1, 2, 4).reshape(b, 20, t, -1)
        e1 = self.conv1(norm_feat)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)

        cnn_out = e6.permute(0, 2, 3, 1).reshape(b, t, -1).permute(0, 2, 1)

        tcn_1, h0 = self.tcn_conv1(cnn_out, hidden[0])
        tcn_2, h1 = self.tcn_conv2(tcn_1, hidden[1])
        tcn_3, h2 = self.tcn_conv3(tcn_2, hidden[2])
        tcn_4, h3 = self.tcn_conv4(tcn_3, hidden[3])
        tcn_5, h4 = self.tcn_conv5(tcn_4, hidden[4])
        tcn_6, h5 = self.tcn_conv6(tcn_5, hidden[5])
        
        tcn_out = torch.stack((tcn_1, tcn_2, tcn_3, tcn_4, tcn_5, tcn_6), dim=1).mean(1)

        lstm_out, lstm_hidden = self.lstm(tcn_out.permute(0, 2, 1), hidden[-1])
        
        dcnn_in = (lstm_out * tcn_out.permute(0, 2, 1)).reshape(b, t, 4, -1).permute(0, 3, 1, 2)
        d6 = self.conv6_t(torch.cat([dcnn_in, e6], dim=1))
        d5 = self.conv5_t(torch.cat([d6, e5], dim=1))
        d4 = self.conv4_t(torch.cat([d5, e4], dim=1))
        d3 = self.conv3_t(torch.cat([d4, e3], dim=1))
        d2 = self.conv2_t(torch.cat([d3, e2], dim=1))
        est_mask = self.conv1_t_mask(torch.cat([d2, e1], dim=1))
        est_mask = est_mask.reshape(b, 2, 1, t, -1).permute(0, 1, 3, 4, 2)
        est_angle = self.conv1_t_angle(torch.cat([d2, e1], dim=1))
        est_angle = est_angle.reshape(b, 2, 2, t, -1).permute(0, 1, 3, 4, 2)
        est_angle = self.angle_norm(est_angle, complex_dim=-1)
        est_vad = torch.sigmoid(self.vad_fc(lstm_out))
        # est_mask = torch.sigmoid(self.mask_fc(d1_mask.squeeze(dim=-1))).unsqueeze(dim=-1)
        # est_angle = torch.tanh(torch.stack([self.real_fc(d1_angle[..., 0]), self.imag_fc(d1_angle[..., 1])], dim=-1))
        mix_mag = (mix_spec ** 2).sum(-1).sqrt().unsqueeze(dim=-1)
        est_spec = mix_mag[:, :, :, 1:] * est_mask * est_angle * est_vad.permute(0, 2, 1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        est_spec = F.pad(est_spec, [0, 0, 1, 0])
        est_wav = self.stft.inverse(est_spec.reshape(b * 2, t, f, -1)).reshape(b, 2, -1)
        if label_speech is not None:
            loss = 0
            for i in range(mix_wav.size(1)):
                sdri_loss = self.sdri(est_wav[:, i], self.stft(label_speech[:, i]), self.stft(mix_wav[:, i])).mean()
                spec_loss = self.spec_loss(est_wav[:, i], self.stft(label_speech[:, i]))
                loss = loss + sdri_loss + spec_loss
            l2_loss = self.l2_regularization(l2_alpha=1)
            features = [e1, e2, e3, e4, e5, tcn_1, tcn_1, tcn_2, tcn_3, tcn_4, tcn_5, tcn_6, d5, d4, d3, d2]
            l2_f_loss = self.l2_regularization_feature(features)
            loss = loss + l2_loss + l2_f_loss
        else:
            loss = None
        return est_wav, loss, (h0.detach(), h1.detach(), h2.detach(), h3.detach(), h4.detach(), h5.detach(), (lstm_hidden[0].detach(), lstm_hidden[1].detach()))

    def l2_regularization(self, l2_alpha=1):
        l2_loss = []
        for module in self.modules():
            if type(module) is nn.Conv2d or type(module) is nn.Conv1d:
                l2_loss.append((module.weight ** 2).mean())
            elif type(module) is nn.LSTM:
                for name, param in module.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        l2_loss.append((param ** 2).mean())
        return l2_alpha * torch.stack(l2_loss, dim=0).mean()
    
    def l2_regularization_feature(self, features, l2_alpha=0.01):
        l2_loss = []
        for f in features:
            l2_loss.append((f ** 2).mean())
        return l2_alpha * torch.stack(l2_loss, dim=0).mean()


    def sdri(self, est, ref, mix):
        target = ref
        noise = target - est
        est_sdr_loss = 10 * torch.log10(torch.sum(target ** 2, 1) / (torch.sum(noise ** 2, 1) + 1.0e-6) + 1.0e-6)

        noise2 = target - mix
        mix_sdr_loss = 10 * torch.log10(torch.sum(target ** 2, 1) / (torch.sum(noise2 ** 2, 1) + 1.0e-6) + 1.0e-6)

        loss = -(est_sdr_loss - mix_sdr_loss)
        return loss
    
    def angle_norm(self, spec, complex_dim=-1):
        mag = ((spec ** 2).sum(complex_dim, keepdims=True) + 1e-7).sqrt()
        norm_spec = spec / (mag + 1e-6)
        return norm_spec
    
    def spec_loss(self, est, ref):
        est_spec = self.stft_for_loss.transform(est)
        lab_spec = self.stft_for_loss.transform(ref)
        r_i_loss = (est_spec - lab_spec).abs()
        mag_loss = (((est_spec ** 2).sum(-1) + 1e-7).sqrt() - ((lab_spec ** 2).sum(-1) + 1e-7).sqrt()).abs()
        loss = torch.cat([r_i_loss, mag_loss.unsqueeze(dim=-1)], dim=-1).mean()
        return loss

def resume_model(net, models_path):
    model_dict = torch.load(models_path, map_location=torch.device('cpu'))
    state_dict = model_dict['state_dict']
    net.load_state_dict(state_dict, False)
    

def gen_net(model_path):
    net = NetCRNN()
    resume_model(net, model_path)
    net.eval()
    return net

if __name__ == '__main__':
    import thop
    import numpy as np
    import soundfile as sf
    
    net = gen_net()
    
    WAV_PATH = './deploy/in_data.wav'
    data, fs = sf.read(WAV_PATH)
    in_data = torch.from_numpy(data.T.astype(np.float32)).unsqueeze(dim=0)
    with torch.no_grad():
        est_wav, _, _ = net(in_data)
    est_wav = est_wav.detach().squeeze().cpu().numpy().T
    sf.write('./deploy/out.wav', est_wav, fs)