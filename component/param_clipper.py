import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import warnings

warnings.filterwarnings("ignore")


class ParamClipperHook(object):
    def __init__(self, k):
        self.k = k
        self.keys = []
        self.is_apply_change = nn.Parameter(torch.tensor(False, dtype=torch.bool), requires_grad=False)

    def compute_weight(self, module):
        for key in self.keys:
            param_new = getattr(module, key['name'] + '_new')
            if not self.is_apply_change:
                with torch.no_grad():
                    if param_new.sum().abs() > 0:
                        self.is_apply_change = True

            if self.is_apply_change:
                delattr(module, key['name'])
                with torch.no_grad():
                    new_k = self.k - 1e-7
                    param_new.data.clamp_(-new_k, new_k)
                param_clip = torch.clamp(param_new, -self.k, self.k)
                setattr(module, key['name'], param_clip)
            else:
                param = getattr(module, key['name'])
                delattr(module, key['name'])
                with torch.no_grad():
                    param_new.data = param.data
                param_clip = torch.clamp(param_new, -self.k, self.k)
                setattr(module, key['name'], param_clip)
        self.is_apply_change = True
        return None

    def __call__(self, module, inputs):
        self.compute_weight(module)

    @staticmethod
    def apply(module, k=0.499):
        fn = ParamClipperHook(k)
        keys = list(module._parameters.keys())
        for key in keys:
            if 'weight' in key or ('bias' in key):
                weight = module._parameters[key]
                fn.keys.append({'name': key, 'shape': weight.size()})
                delattr(module, key)
                param_new = Parameter(torch.zeros_like(weight))
                module.register_parameter(key + '_new', param_new)
                weight.requires_grad = False
                setattr(module, key, weight)
        module.register_forward_pre_hook(fn)
        return module
