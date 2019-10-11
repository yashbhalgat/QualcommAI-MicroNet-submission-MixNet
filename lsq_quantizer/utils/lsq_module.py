import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import pdb

class LsqWeightFun(autograd.Function):
    def __init__(self, constraint, skip_bit=None):
        super(LsqWeightFun, self).__init__()
        self.valmin = float(constraint.min())
        self.valmax = float(constraint.max())
        self.skip_bit = skip_bit

    def forward(self, *args, **kwargs):
        x = args[0]
        scale = args[1]
        skip_bit = self.skip_bit
        x_scale = torch.div(x, scale)
        x_clip = F.hardtanh(x_scale, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        if skip_bit:
            sign = x_round.sign()
            x_round = torch.floor(torch.abs(x_round)/2**skip_bit)*(2**skip_bit)
            x_round = torch.mul(x_round, sign)
        
        x_restore = torch.mul(x_round, scale)
        self.save_for_backward(x_clip)
        return x_restore

    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        x_clip = self.saved_tensors[0]
        internal_flag = ((x_clip > self.valmin) - (x_clip >= self.valmax)).float()

        # gradient for weight
        grad_weight = grad_top

        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_scale = (grad_scale_elem * grad_top).sum().view((1,))
        return grad_weight, grad_scale


class LsqWeight(nn.Module):
    def __init__(self, constraint, scale_init=None, skip_bit=None):
        super(LsqWeight, self).__init__()
        self.constraint = constraint
        self.skip_bit = skip_bit
        scale_init = scale_init if scale_init is not None else torch.ones(1)
        self.scale = nn.Parameter(scale_init)

    def extra_repr(self):
        return 'constraint=%s' % self.constraint

    def forward(self, x):
        wquantizer = LsqWeightFun(self.constraint, skip_bit=self.skip_bit)
        return wquantizer(x, self.scale)


class LsqActivationFun(autograd.Function):
    def __init__(self, constraint, skip_bit=None):
        super(LsqActivationFun, self).__init__()
        self.valmin = float(constraint.min())
        self.valmax = float(constraint.max())
        self.skip_bit = skip_bit

    def forward(self, *args, **kwargs):
        x = args[0]
        scale = args[1]
        skip_bit = self.skip_bit
        x_scale = torch.div(x, scale)
        x_clip = F.hardtanh(x_scale, min_val=self.valmin, max_val=self.valmax)
        x_round = torch.round(x_clip)
        if skip_bit:
            sign = x_round.sign()
            x_round = torch.floor(torch.abs(x_round)/2**skip_bit)*(2**skip_bit)
            x_round = torch.mul(x_round, sign)
        
        x_restore = torch.mul(x_round, scale)
        self.save_for_backward(x_clip)
        return x_restore

    def backward(self, *grad_outputs):
        grad_top = grad_outputs[0]
        x_clip = self.saved_tensors[0]
        internal_flag = ((x_clip > self.valmin) - (x_clip >= self.valmax)).float()

        # gradient for activation
        grad_activation = grad_top * internal_flag

        # gradient for scale
        grad_one = x_clip * internal_flag
        grad_two = torch.round(x_clip)
        grad_scale_elem = grad_two - grad_one
        grad_scale = (grad_scale_elem * grad_top).sum().view((1,))
        return grad_activation, grad_scale


class LsqActivation(nn.Module):
    def __init__(self, constraint, scale_init=None, skip_bit=None):
        super(LsqActivation, self).__init__()
        self.constraint = constraint
        scale_init = scale_init if scale_init is not None else torch.ones(1)
        self.scale = nn.Parameter(scale_init)
        self.skip_bit = skip_bit

    def extra_repr(self):
        return 'constraint=%s' % self.constraint

    def forward(self, x):
        aquantizer = LsqActivationFun(self.constraint, skip_bit=self.skip_bit)
        return aquantizer(x, self.scale)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,stride,
                                     padding, dilation, groups, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.wquantizer = None

    def forward(self, x):
        weight = self.weight if self.wquantizer is None else self.wquantizer(self.weight)
        return F.linear(x, weight, self.bias)


@autograd.no_grad()
def add_lsqmodule(net, constr_weight):
    for name, module in net.named_modules():
        if isinstance(module, Conv2d) or isinstance(module, Linear):
            scale_init = torch.full((1,), module.weight.abs().mean().item())
            module.wquantizer = LsqWeight(constraint=constr_weight, scale_init=scale_init.clone())
