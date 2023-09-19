import torch


class DeepFilter():
    def __init__(self, L, I, is_look_forward=False):
        self.time_pad = L
        self.feat_pad = I
        self.is_look_forward = is_look_forward

    def deep_filter_data(self, data, device=None):
        bt, t, f, ch = data.size()
        if device is None:
            device = data.device
        if self.is_look_forward:
            pad_in = torch.cat([torch.zeros([bt, self.time_pad, f, ch], device=device), data,
                                torch.zeros([bt, self.time_pad, f, ch], device=device)], dim=1)
        else:
            pad_in = torch.cat([torch.zeros([bt, self.time_pad * 2, f, ch], device=device), data], dim=1)
        pad_in = torch.cat([torch.zeros([bt, t + self.time_pad * 2, self.feat_pad, ch], device=device), pad_in,
                            torch.zeros([bt, t + self.time_pad * 2, self.feat_pad, ch], device=device)], dim=2)

        # slice time dim
        t_list = []
        for i in range(2 * self.time_pad + 1):
            t_list.append(pad_in[:, i:i + t])
        stack_res = torch.stack(t_list, dim=1)

        # slice feat dim
        f_list = []
        for i in range(2 * self.feat_pad + 1):
            f_list.append(stack_res[:, :, :, i:i + f])
        stack_res = torch.stack(f_list, dim=5).permute(0, 1, 5, 4, 2, 3)
        return stack_res
