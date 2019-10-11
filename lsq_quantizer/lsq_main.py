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

import pdb
from helpers import load_checkpoint

def main():
    os.chdir(os.path.dirname(__file__))
    args = get_arguments()
    constr_weight = get_constraint(args.weight_bits, 'weight')
    constr_activation = get_constraint(args.activation_bits, 'activation')
    if args.dataset == 'cifar10':
        network = resnet20
        dataloader = dataloader_cifar10
    elif args.dataset == 'cifar100':
        t_net = ResNet(depth=56, num_classes=100)
        state = torch.load("/prj/neo_lv/user/ybhalgat/LSQ-KD/cifar100_pretrained/resnet56.pth.tar")
        t_net.load_state_dict(state)
        network = resnet20
        dataloader = dataloader_cifar100
    else:
        if args.network == 'resnet18':
            network = resnet18
        elif args.network == 'resnet50':
            network = resnet50
        elif args.network == 'efficientnet-b0':
            t_net = EfficientNet.from_pretrained("efficientnet-b3")
            network = efficientnet_b0
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

    model_path = os.path.join(args.model_root, args.model_name + '.pth.tar')
    if not os.path.exists(model_path):
        model_path = model_path[:-4]
    name_weights_old = torch.load(model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    load_checkpoint(net, name_weights_new)
    # net.load_state_dict(name_weights_new, strict=False)
    if not args.haq:
        add_lsqmodule(net, bit_width=args.weight_bits)
    else:
        if args.network == 'resnet50':
            strategy = [6, 6, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 3, 5, 4, 3, 5, 4, 3, 4, 4, 4, 2, 5,
                        4, 3, 3, 5, 3, 2, 5, 3, 2, 4, 3, 2, 5, 3, 2, 5, 3, 4, 2, 5, 2, 3, 4, 2, 3, 4]
        elif args.network == 'efficientnet-b0':
            strategy = [7, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6,
                        6, 6, 6, 6, 6, 7, 6, 7, 6, 7, 6, 5, 6, 5, 6, 4, 5, 6, 5, 6, 4, 4, 5, 4, 5, 2,
                        3, 4, 3, 4, 2, 3, 4, 4, 7, 5, 2, 4, 2, 5, 5, 2, 4, 2, 5, 5, 2, 4, 2, 5, 5, 2,
                        4, 3, 3, 2]
        add_lsqmodule(net, strategy=strategy)

    print(net)
    net = net.cuda()
    net = nn.DataParallel(net, device_ids=range(cuda.device_count()))

    t_net = t_net.cuda()
    t_net = nn.DataParallel(t_net, device_ids=range(cuda.device_count()))



    quan_activation = isinstance(constr_activation, np.ndarray)
    postfix = '_w' if not quan_activation else '_a'
    new_model_name = args.prefix + args.model_name + '_lsq' + postfix
    cache_root = os.path.join('.', 'cache')
    train_loger = LogHelper(new_model_name, cache_root, quan_activation, args.resume)
    optimizer, lr_scheduler, optimizer_t = get_optimizer(s_net=net,
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
                      model_name=new_model_name,
                      train_loger=train_loger)
    trainer(total_epoch=args.total_epoch,
            save_check_point=True,
            resume=args.resume)


if __name__ == '__main__':
    main()



