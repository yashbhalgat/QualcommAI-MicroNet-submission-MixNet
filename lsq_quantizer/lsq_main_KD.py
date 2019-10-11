import os
import numpy as np
import torch
import torch.nn as nn
import torch.cuda as cuda
from utils.data_loader import dataloader_cifar10, dataloader_cifar100
from utils.data_loader import dataloader_imagenet
from utils.lsq_train import LogHelper
from utils.lsq_train import get_arguments
from utils.lsq_train import get_optimizer
from utils.add_lsqmodule import add_lsqmodule
from utils.lsq_network import resnet18
from utils.lsq_network import resnet20
from utils.lsq_network import resnet50
from utils.effnet import efficientnet_b0
from utils.effnet import EfficientNet
from utils.utilities import Trainer
from utils.utilities import get_constraint
from utils.resnet import ResNet
from utils.wrn import WRN40_4, WRN40_6
from utils.mixnet import mixnet_s
from utils.mixnet_FP import MixNet


import pdb
from helpers import load_checkpoint
from utils.utilities import get_constraint, start_LSQ, count_sparsity, make_weights_zero


class _CustomDataParallel(nn.DataParallel):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(_CustomDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main():
    os.chdir(os.path.dirname(__file__))
    args = get_arguments()
    constr_weight = get_constraint(args.weight_bits, 'weight')
    constr_activation = get_constraint(args.activation_bits, 'activation')
    if args.dataset == 'cifar10':
        network = resnet20
        dataloader = dataloader_cifar10
    elif args.dataset == 'cifar100':
        t_net = WRN40_6()
        state = torch.load("/prj/neo_lv/user/ybhalgat/LSQ-KD-0911/cifar100_pretrained/wrn40_6.pth")
        t_net.load_state_dict(state)
        network = WRN40_4
        dataloader = dataloader_cifar100
    else:
        if args.network == 'resnet18':
            network = resnet18
        elif args.network == 'resnet50':
            network = resnet50
        elif args.network == 'efficientnet-b0':
            t_net = EfficientNet.from_pretrained("efficientnet-b1")
            network = efficientnet_b0
        elif args.network == "mixnet_s":
            t_net = MixNet(net_type=args.teacher)
            t_net.load_state_dict(torch.load("../imagenet_pretrained/"+args.teacher+".pth"))
            network = mixnet_s
        else:
            print('Not Support Network Type: %s' % args.network)
            return
        dataloader = dataloader_imagenet
    train_loader = dataloader(args.data_root, split='train', batch_size=args.batch_size)
    test_loader = dataloader(args.data_root, split='test', batch_size=args.batch_size)
    net = network(quan_first=args.quan_first,
                  quan_last=args.quan_last,
                  constr_activation=constr_activation,
                  preactivation=args.preactivation,
                  bw_act=args.activation_bits)

    # net.load_state_dict(name_weights_new, strict=False)
    if args.cem:
        ##### CEM vector for 1.5x_W7A7_CEM prefinetuning 72%
        # cem_input = [7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7,
        #              7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5,
        #              7, 7, 7, 6, 7, 7, 7, 7, 7, 5, 7, 7, 7, 6, 4, 7, 7, 6,
        #              6, 6, 7, 7, 7, 7, 5, 7, 7, 7, 6, 4, 7, 7, 5, 5, 4, 7,
        #              7, 6, 5, 5, 7, 5, 7, 5, 5, 3]

        ##### CEM vector for 1.5x_W7A7_CEM prefinetuning 70%
        cem_input = [7, 7, 7, 7, 7, 5, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7,
                     6, 7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 7, 6, 7, 7, 7, 7, 4,
                     7, 6, 7, 5, 7, 7, 7, 7, 7, 5, 7, 7, 7, 5, 5, 7, 7, 7,
                     5, 6, 7, 7, 7, 6, 4, 7, 7, 6, 5, 4, 7, 6, 5, 5, 4, 7,
                     7, 6, 5, 4, 7, 7, 6, 5, 5, 3]

        strategy_path = "/prj/neo_lv/user/ybhalgat/LSQ-implementation/lsq_quantizer/cem_strategy_relaxed.txt"
        with open(strategy_path) as fp:
            strategy = fp.readlines()
        strategy = [x.strip().split(",") for x in strategy]

        ##### CEM vector for W6A6_CEM
        # cem_input = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        #              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
        #              0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 
        #              1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 
        #              1, 0, 1, 1, 0, 1, 1, 1, 1, 1]

        strat = {}
        act_strat = {}
        for idx, width in enumerate(cem_input):
            weight_layer_name = strategy[idx][1]
            act_layer_name = strategy[idx][0]
            for name, module in net.named_modules():
                if name.startswith('module'):
                    name = name[7:]  # remove `module.`
                if name==weight_layer_name:
                    strat[name] = int(cem_input[idx])
                if name==act_layer_name:
                    act_strat[name] = int(cem_input[idx])

        add_lsqmodule(net, bit_width=args.weight_bits, strategy=strat)

        for name, module in net.named_modules():
            if name in act_strat:
                if "efficientnet" in args.network:
                    if "_in_act_quant" in name or "first_act" in name or "_head_act_quant0" in name or "_head_act_quant1" in name:
                        temp_constr_act = get_constraint(act_strat[name], 'weight') #symmetric
                    else:
                        temp_constr_act = get_constraint(act_strat[name], 'activation') #asymmetric
                elif "mixnet" in args.network:
                    if "last_act" in name or "out_act_quant" in name or "first_act" in name:
                        temp_constr_act = get_constraint(act_strat[name], 'weight') #symmetric
                    else:
                        temp_constr_act = get_constraint(act_strat[name], 'activation') #asymmetric
                module.constraint = temp_constr_act

    elif args.manual:
        if args.network == "wrn40_4":
            strategy = {"block3.layer.0.conv2": 3, 
                        "block3.layer.2.conv1": 3,
                        "block3.layer.3.conv1": 3,
                        "block3.layer.4.conv1": 3,
                        "block3.layer.2.conv2": 3,
                        "block3.layer.1.conv2": 3,
                        "block3.layer.3.conv2": 3,
                        "block3.layer.1.conv1": 3,
                        "block3.layer.5.conv1": 2,
                        "block1.layer.1.conv2": 1}
            act_strategy = {"block3.layer.0.relu2": 3,
                            "block3.layer.2.relu1": 3,
                            "block3.layer.3.relu1": 3,
                            "block3.layer.4.relu1": 3,
                            "block3.layer.2.relu2": 3,
                            "block3.layer.1.relu2": 3,
                            "block3.layer.3.relu2": 3,
                            "block3.layer.1.relu1": 3,
                            "block3.layer.5.relu1": 2,
                            "block1.layer.1.relu2": 1}

        elif args.network == 'efficientnet-b0':
            strategy = {"_fc": 3,
                        "_conv_head": 5,
                        "_blocks.15._project_conv": 5,
                        "_blocks.15._expand_conv": 4,
                        "_blocks.14._expand_conv": 4,
                        "_blocks.13._expand_conv": 4,
                        "_blocks.12._expand_conv": 4,
                        "_blocks.13._project_conv": 4,
                        "_blocks.14._project_conv": 4,
                        "_blocks.12._project_conv": 5,
                        "_blocks.9._expand_conv": 4,
                        "_blocks.10._expand_conv": 4}
            act_strategy = {"_head_act_quant1": 3,
                            "_head_act_quant0": 5,
                            "_blocks.15._pre_proj_activation": 5,
                            "_blocks.15._in_act_quant": 4,
                            "_blocks.14._in_act_quant": 4,
                            "_blocks.13._in_act_quant": 4,
                            "_blocks.12._in_act_quant": 4,
                            "_blocks.13._pre_proj_activation": 4,
                            "_blocks.14._pre_proj_activation": 4,
                            "_blocks.12._pre_proj_activation": 5,
                            "_blocks.9._in_act_quant": 4,
                            "_blocks.10._in_act_quant": 4}
            #strategy = {"_fc": 3,
            #            "_conv_head": 4,
            #            "_blocks.15._project_conv": 4,
            #            "_blocks.14._project_conv": 4,
            #            "_blocks.13._project_conv": 3,
            #            "_blocks.13._expand_conv": 4,
            #            "_blocks.12._project_conv": 4,
            #            "_blocks.12._expand_conv": 5,
            #            "_blocks.14._expand_conv": 4,
            #            "_blocks.15._expand_conv": 4,
            #            "_blocks.9._project_conv": 4}
            #            #"_blocks.10._project_conv": 4,
            #            #"_blocks.9._expand_conv": 4,
            #            #"_blocks.10._expand_conv": 4,
            #            #"_blocks.7._expand_conv": 4,
            #            #"_blocks.11._expand_conv": 4}
            #act_strategy = {"_head_act_quant1": 3,
            #                "_head_act_quant0": 4,
            #                "_blocks.15._pre_proj_activation": 4,
            #                "_blocks.14._pre_proj_activation": 4,
            #                "_blocks.13._pre_proj_activation": 3,
            #                "_blocks.13._in_act_quant": 4,
            #                "_blocks.12._pre_proj_activation": 4,
            #                "_blocks.12._in_act_quant": 5,
            #                "_blocks.14._in_act_quant": 4,
            #                "_blocks.15._in_act_quant": 4,
            #                "_blocks.9._pre_proj_activation": 4}
            #                #"_blocks.10._pre_proj_activation": 4,
            #                #"_blocks.9._in_act_quant": 4,
            #                #"_blocks.10._in_act_quant": 4,
            #                #"_blocks.7._in_act_quant": 4,
            #                #"_blocks.11._in_act_quant": 4}
        add_lsqmodule(net, bit_width=args.weight_bits, strategy=strategy)
        
        for name, module in net.named_modules():
            if name in act_strategy:
                if "_in_act_quant" in name or "first_act" in name or "_head_act_quant0" in name or "_head_act_quant1" in name:
                    temp_constr_act = get_constraint(act_strategy[name], 'weight') #symmetric
                else:
                    temp_constr_act = get_constraint(act_strategy[name], 'activation') #asymmetric
                module.constraint = temp_constr_act

    elif args.haq:
        if args.network == 'resnet50':
            strategy = [6, 6, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 3, 5, 4, 3, 5, 4, 3, 4, 4, 4, 2, 5,
                        4, 3, 3, 5, 3, 2, 5, 3, 2, 4, 3, 2, 5, 3, 2, 5, 3, 4, 2, 5, 2, 3, 4, 2, 3, 4]
        elif args.network == 'efficientnet-b0':
            strategy = [7, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6,
                        6, 6, 6, 6, 6, 7, 6, 7, 6, 7, 6, 5, 6, 5, 6, 4, 5, 6, 5, 6, 4, 4, 5, 4, 5, 2,
                        3, 4, 3, 4, 2, 3, 4, 4, 7, 5, 2, 4, 2, 5, 5, 2, 4, 2, 5, 5, 2, 4, 2, 5, 5, 2,
                        4, 3, 3, 2]
        add_lsqmodule(net, strategy=strategy)

    else:
        add_lsqmodule(net, bit_width=args.weight_bits)


    model_path = os.path.join(args.model_root, args.model_name + '.pth.tar')
    if not os.path.exists(model_path):
        model_path = model_path[:-4]
    name_weights_old = torch.load(model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    load_checkpoint(net, name_weights_new, strict=False)

    print(net)
    net = net.cuda()
    net = nn.DataParallel(net)

    t_net = t_net.cuda()
    t_net = nn.DataParallel(t_net)

    if args.pruned:
        start_LSQ(net)

    quan_activation = isinstance(constr_activation, np.ndarray)
    postfix = '_w' if not quan_activation else '_a'
    new_model_name = args.prefix + args.model_name + '_lsq' + postfix
    cache_root = os.path.join('.', 'cache')
    train_loger = LogHelper(new_model_name, cache_root, quan_activation, args.resume)
    optimizer, lr_scheduler, optimizer_t,lr_scheduler_t = get_optimizer(s_net=net,
                                            t_net=t_net,
                                            optimizer=args.optimizer,
                                            lr_base=args.learning_rate,
                                            weight_decay=args.weight_decay,
                                            lr_scheduler=args.lr_scheduler,
                                            total_epoch=args.total_epoch,
                                            quan_activation=quan_activation,
                                            act_lr_factor=args.act_lr_factor,
                                            weight_lr_factor=args.weight_lr_factor)
    trainer = Trainer(net=net,
                      t_net=t_net,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      optimizer_t=optimizer_t,
                      lr_scheduler=lr_scheduler,
                      lr_scheduler_t=lr_scheduler_t,
                      model_name=new_model_name,
                      train_loger=train_loger,
                      pruned=args.pruned)
    trainer(total_epoch=args.total_epoch,
            save_check_point=True,
            resume=args.resume)


if __name__ == '__main__':
    main()



