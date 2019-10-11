import time
from cem import CEM
import argparse
import os
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.cuda as cuda
from utils.data_loader import dataloader_imagenet
from utils.lsq_train import LogHelper
from utils.lsq_train import get_optimizer
from utils.add_lsqmodule import add_lsqmodule
from utils.lsq_network import resnet18
from utils.lsq_network import resnet20
from utils.lsq_network import resnet50
from utils.effnet import efficientnet_b0
from utils.utilities import Trainer
from utils.utilities import get_constraint, eval_performance
from flops_counter import add_flops_counting_methods, get_model_parameters_number
import pdb
from helpers import load_checkpoint

def cem_eval(cem_input, data_root="/nvme/users/tijmen/imagenet/", batch_size=350,
             strategy_path="/prj/neo_lv/user/ybhalgat/LSQ-implementation/lsq_quantizer/cem_strategy_relaxed.txt", 
             model_path="/prj/neo_lv/scratch/jinwonl/finetune/lsq_quantizer/models/1.5x_W8A8_CEM_re1/efficientnet-b0.pth",
             activation_bits=7, weight_bits=7):
    
    constr_activation = get_constraint(activation_bits, 'activation')
    network = efficientnet_b0
    dataloader = dataloader_imagenet

    test_loader = dataloader(data_root, split='test', batch_size=batch_size)
    with open(strategy_path) as fp:
        strategy = fp.readlines()
    strategy = [x.strip().split(",") for x in strategy]

    net = network(quan_first=True,
                  quan_last=True,
                  constr_activation=constr_activation,
                  preactivation=False,
                  bw_act=activation_bits)
    
    strat = {}
    act_strat = {}
    for idx, flag in enumerate(cem_input):
        weight_layer_name = strategy[idx][1]
        act_layer_name = strategy[idx][0]
        for name, module in net.named_modules():
            if name.startswith('module'):
                name = name[7:]  # remove `module.`
            if name==weight_layer_name:
                strat[name] = int(cem_input[idx])
            if name==act_layer_name:
                act_strat[name] = int(cem_input[idx])

    add_lsqmodule(net, bit_width=weight_bits, strategy=strat)

    for name, module in net.named_modules():
        if name in act_strat:
            if "_in_act_quant" in name or "first_act" in name or "_head_act_quant0" in name or "_head_act_quant1" in name:
                temp_constr_act = get_constraint(act_strat[name], 'weight') #symmetric
            else:
                temp_constr_act = get_constraint(act_strat[name], 'activation') #asymmetric
            module.constraint = temp_constr_act
    
    net.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()

    # Calculate score
    flops_model = add_flops_counting_methods(net)
    flops_model.eval().start_flops_count()
    input_res = (3, 224, 224)
    batch = torch.ones(()).new_empty((1, *input_res),
                                     dtype=next(flops_model.parameters()).dtype,
                                     device=next(flops_model.parameters()).device)
    _ = flops_model(batch)
    flops_count = flops_model.compute_average_flops_cost(bw_weight=activation_bits,
                                                         bw_act=weight_bits,
                                                         strategy=(strat, strat))
    params_count = get_model_parameters_number(flops_model, bw_weight=weight_bits, w_strategy=strat)
    flops_model.stop_flops_count()

    score = params_count/6900000.0 + flops_count/1170000000.0


    # Calculate accuracy
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=range(cuda.device_count()))

    quan_perf_epoch = eval_performance(net, test_loader, criterion)
    accuracy = quan_perf_epoch[1]

    return accuracy, score

####################################################################3
# cem test
num_actions = 82 # [number of layers]
num_pop = 100 # [100, 200, 500]
elite_fraction = 0.1 #[0.05, 0.1, 0.2]
num_gen = 20 #[30, 100, 200, 500]
bias = 0.5

# for reproducibility
np.random.seed(0)

#cem = CEMBinary(num_actions, num_pop, elite_fraction)
cem_init = [7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7,
            7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5,
            7, 7, 7, 6, 7, 7, 7, 7, 7, 5, 7, 7, 7, 6, 4, 7, 7, 6,
            6, 6, 7, 7, 7, 7, 5, 7, 7, 7, 6, 4, 7, 7, 5, 5, 4, 7,
            7, 6, 5, 5, 7, 5, 7, 5, 5, 3]

cem = CEM(num_actions, num_pop, elite_fraction, bias, cem_init)
max_reward = 0

# generation loop
for i in range(0, num_gen):
    # population loop
    for k in range(0, num_pop):
        start = time.time()
        inst_action = cem.make_and_act_policy(k)
        acc, score = cem_eval(inst_action)

        inst_reward = 0
        acc_reward = 0
        if (acc >= 72):
            acc_reward = 1
        elif (acc >= 65):
            acc_reward = (acc - 65)/(72.0 - 65.0)
        
        score_reward = 0
        if (score < 0.0):
            score_reward = 1
        elif (score < 0.195):
            score_reward = 1 - (score - 0.0)/(0.195 - 0.0)

        inst_reward = acc_reward * score_reward

        print("{} {} {:.2f} {:.4f} {:.4f} {:.4f} {:.4f} {:.0f} {:.0f}s".format(i, k, acc, score, acc_reward, score_reward, inst_reward, 
            sum(inst_action), time.time() - start))
        
        if (inst_reward > max_reward):
            max_reward = inst_reward
            print("== MAX: {} {} {:.2f} {:.4f} {:.4f} {:.0f} | {} {} {} | {}".format(i, k, acc, score, max_reward, 
                sum(inst_action), num_gen, num_pop, bias, inst_action.astype(int).tolist()))
        
        cem.reward(k, inst_reward)
    
    # learn next generation
    cem.learn(i)
