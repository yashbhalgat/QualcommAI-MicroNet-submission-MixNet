import torch
import torch.autograd as autograd

from .lsq_module import Conv2d, Linear, LsqWeight
from .effnet_utils import Conv2dSamePadding
from .utilities import get_constraint
import pdb

@autograd.no_grad()
def add_lsqmodule(net, bit_width=4, strategy=None, skip_name=None, skip_bit=None):
    bit_width_map = {}
    quant_layer_idx = 0
    for name, module in net.named_modules():
        if isinstance(module, Conv2d) or isinstance(module, Linear):
            if strategy==None:
                bit_width_map[name] = get_constraint(bit_width, "weight")
            elif isinstance(strategy, list):
                bit_width_map[name] = get_constraint(strategy[quant_layer_idx], "weight")
            elif isinstance(strategy, dict):
                if name in strategy:
                    bit_width_map[name] = get_constraint(strategy[name], "weight")
                else:
                    bit_width_map[name] = get_constraint(bit_width, "weight")
            quant_layer_idx += 1
        
        elif isinstance(module, Conv2dSamePadding):
            if module.quantize_w:
                if strategy==None:
                    bit_width_map[name] = get_constraint(bit_width, "weight")
                elif isinstance(strategy, list):
                    bit_width_map[name] = get_constraint(strategy[quant_layer_idx], "weight")
                elif isinstance(strategy, dict):
                    if name in strategy:
                        bit_width_map[name] = get_constraint(strategy[name], "weight")
                    else:
                        bit_width_map[name] = get_constraint(bit_width, "weight")
                quant_layer_idx += 1
        

    layer_name = None
    for name, module in net.named_modules():
        if skip_name==name:
            sb = skip_bit
        else:
            sb = None

        if isinstance(module, Conv2d) or isinstance(module, Linear):
            if sb:
                layer_name = name
            scale_init = torch.full((1,), module.weight.abs().mean().item())
            module.wquantizer = LsqWeight(constraint=bit_width_map[name], scale_init=scale_init.clone(), skip_bit=sb)
            quant_layer_idx += 1

        elif isinstance(module, Conv2dSamePadding):
            if module.quantize_w:
                if sb:
                    layer_name = name
                scale_init = torch.full((1,), module.weight.abs().mean().item())
                module.wquantizer = LsqWeight(constraint=bit_width_map[name], scale_init=scale_init.clone(), skip_bit=sb)
                quant_layer_idx += 1
    
    return layer_name
