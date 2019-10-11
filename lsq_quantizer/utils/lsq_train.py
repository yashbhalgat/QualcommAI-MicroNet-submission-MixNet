import os
import argparse
import torch.optim as optim


def get_arguments():
    parser = argparse.ArgumentParser(description='LSQ Training')
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--teacher', required=False, type=str)
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'cifar100', 'imagenet'], type=str)
    parser.add_argument('--network', required=True, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--model_root', default=None, type=str)

    parser.add_argument('--weight_bits', required=True,  type=int)
    parser.add_argument('--activation_bits', default=0, type=int)
    parser.add_argument('--preactivation', default=False, action='store_true')
    parser.add_argument('--quan_first', default=False, action='store_true')
    parser.add_argument('--quan_last', default=False, action='store_true')

    parser.add_argument('--total_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--act_lr_factor', default=10, type=float)
    parser.add_argument('--weight_lr_factor', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', type=str)

    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--prefix', default='', type=str)
   
    parser.add_argument('--cem', default=False, action='store_true',
                        help='use cem-based bit-widths')
    parser.add_argument('--manual', default=False, action='store_true',
                        help='use manual bit-widths')
    parser.add_argument('--haq', default=False, action='store_true',
                        help='to use variable bit-widths obtained from HAQ or not')
    parser.add_argument('--pruned', default=False, action='store_true',
                        help='flag to quantize pruned model')

    args = parser.parse_args()
    if args.data_root is None:
        if args.dataset == 'cifar10':
            args.data_root = os.path.join('F:', '0data', 'cifar')
        else:
            args.data_root = os.path.join('/', 'share', 'buyingjia', 'common_dataset', 'ImageNetOrigin')
    if args.model_root is None:
        if 'cifar10' == args.dataset:
            args.model_root = os.path.join('F:', '1pretrain_model')
        else:
            args.model_root = os.path.join('/', 'data', '1pretrain_model')

    print(' ... Experiment Setting ...')
    print('model_name=%s' % args.model_name)
    print('dataset=%s' % args.dataset)
    print('network=%s' % args.network)
    print('data_root=%s' % args.data_root)
    print('model_root=%s' % args.model_root)
    print('--------------------')

    print('weight_bits=%s' % args.weight_bits)
    print('activation_bits=%s' % args.activation_bits)
    print('preactivation=%s' % args.preactivation)
    print('quan_first=%s' % args.quan_first)
    print('quan_last=%s' % args.quan_last)
    print('--------------------')

    print('optimizer=%s' % args.optimizer)
    print('batch_size=%s' % args.batch_size)
    print('total_epoch=%s' % args.total_epoch)
    print('learning_rate=%s' % args.learning_rate)
    print('weight_decay=%s' % args.weight_decay)
    print('lr_scheduler=%s' % args.lr_scheduler)
    print('--------------------')

    print('resume=%s' % args.resume)
    print('prefix=%s' % args.prefix)
    return args


def get_optimizer(s_net, t_net, optimizer, lr_base, weight_decay, lr_scheduler, total_epoch, quan_activation=False,
        act_lr_factor=10, weight_lr_factor=0.01):
    weight_bias = list()
    scale_weight = list()
    scale_activation = list()
    for name, param in s_net.named_parameters():
        if 'wquantizer' in name and 'scale' in name:
            scale_weight.append(param)
        elif 'activation' in name and 'scale' in name:
            scale_activation.append(param)
        else:
            weight_bias.append(param)

    if not quan_activation:
        param_groups1 = [{'params': weight_bias, 'lr': lr_base}]
        param_groups2 = [{'params': scale_weight, 'lr': lr_base*weight_lr_factor}]
    else:
        param_groups1 = [{'params': weight_bias, 'lr': lr_base}]
        param_groups2 = [{'params': scale_weight, 'lr': lr_base*weight_lr_factor},
                        {'params': scale_activation, 'lr': lr_base*act_lr_factor}]

    if optimizer == 'adam':
        param_groups = param_groups1 + param_groups2
        optimizer = optim.Adam(param_groups, lr=lr_base, weight_decay=weight_decay)
        optimizer_t = optim.Adam(t_net.parameters(), lr=lr_base, weight_decay=weight_decay)

    else:
        param_groups = param_groups1 + param_groups2
        optimizer = optim.SGD(param_groups, lr=lr_base, momentum=0.9, weight_decay=weight_decay)
        optimizer_t = optim.SGD(t_net.parameters(), lr=lr_base, momentum=0.9, weight_decay=weight_decay)


    if lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=1e-11)
        lr_scheduler_t = optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=total_epoch, eta_min=1e-11)


    elif lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
        lr_scheduler_t = optim.lr_scheduler.StepLR(optimizer_t, 10, gamma=0.1, last_epoch=-1)


    else:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        lr_scheduler_t = optim.lr_scheduler.ExponentialLR(optimizer_t, gamma=0.9)

    return optimizer, lr_scheduler, optimizer_t, lr_scheduler_t


class LogHelper:
    def __init__(self, model_name, cache_root=None, quan_activation=False, resume=False):
        self.model_name = model_name
        self.log_cache = ''
        self.ckpt_cache = ''
        self.model_cache = ''
        self.prepare_cache_root(cache_root)
        self.clear_cache_root(resume)

        self.trainlog_path = os.path.join(self.log_cache, model_name + '_trainlog.txt')
        self.perf_path = os.path.join(self.log_cache, model_name + '_perf.txt')
        self.scale_w_path = os.path.join(self.log_cache, model_name + '_scale_weight.txt')
        self.scale_a_path = None
        if quan_activation:
            self.scale_a_path = os.path.join(self.log_cache, model_name + '_scale_activation.txt')

    def prepare_cache_root(self, cache_root=None):
        if cache_root is None or not os.path.exists(cache_root):
            cache_root = os.path.join('.', 'cache')
            if not os.path.exists('./cache'):
                os.mkdir('./cache')
        self.ckpt_cache = os.path.join(cache_root, 'ckpt_cache')
        if not os.path.exists(self.ckpt_cache):
            os.mkdir(self.ckpt_cache)
        self.model_cache = os.path.join(cache_root, 'model_cache')
        if not os.path.exists(self.model_cache):
            os.mkdir(self.model_cache)
        self.log_cache = os.path.join(cache_root, 'log_cache')
        if not os.path.exists(self.log_cache):
            os.mkdir(self.log_cache)

    def clear_cache_root(self, resume):
        if resume:
            return None
        def clear_root(root, model_name):
            for file in os.listdir(root):
                if model_name in file:
                    os.remove(os.path.join(root, file))
        clear_root(self.log_cache, self.model_name)
        clear_root(self.ckpt_cache, self.model_name)
        clear_root(self.model_cache, self.model_name)

    def print_log(self, index, net, msg):
        with open(self.trainlog_path, 'a') as fd:
            fd.write(msg + '\n')

        if self.scale_w_path is not None and index % 100 == 0:
            info = ''
            state_dict = net.state_dict()
            for name in state_dict.keys():
                if 'wquantizer' in name and 'scale' in name:
                    info += '%.10f, ' % state_dict[name].item()
            info = info[:-2] + '\n'
            with open(self.scale_w_path, 'a') as fd:
                fd.write(info)

        if self.scale_a_path is not None and index % 100 == 0:
            info = ''
            state_dict = net.state_dict()
            for name in state_dict.keys():
                if 'activation' in name and 'scale' in name:
                    info += '%.10f, ' % state_dict[name].item()
            info = info[:-2] + '\n'
            with open(self.scale_a_path, 'a') as fd:
                fd.write(info)

    def print_perf(self, perf):
        info = ''
        for item in perf:
            info += '%.4f, ' % item
        info = info[:-2] + '\n'
        with open(self.perf_path, 'a') as fd:
            fd.write(info)

