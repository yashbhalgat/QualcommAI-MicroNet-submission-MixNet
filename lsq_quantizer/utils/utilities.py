import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from datetime import datetime
import torch.cuda as cuda
from torch.autograd import Variable
import torch.nn.functional as F
import pdb


def get_all_layer_names(model, subtypes=None):
    if subtypes is None:
        return [name for name, module in model.named_modules()][1:]
    return [name for name, module in model.named_modules() if isinstance(module, subtypes)]

def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None

def get_layer_name(model, layer):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


def count_sparsity(model):
    total_num_weights = 0
    layer_names = get_all_layer_names(model, (torch.nn.Conv2d, torch.nn.Linear))
    for layer_name in layer_names:
        module = get_layer_by_name(model, layer_name)
        total_num_weights += module.weight.data.numel()

    weights = torch.zeros(total_num_weights).cuda()
    idx = 0
    for layer_name in layer_names:
        module = get_layer_by_name(model, layer_name)
        num_weights = module.weight.data.numel()
        weights[idx:idx+num_weights] = module.weight.data.view(-1).abs().clone()
        idx += num_weights

    num_pruned_weights = weights.abs().eq(0.0).float().sum()
    print('Total # params: {}, pruned # params: {}, pruning ratio: {}'.format(
        total_num_weights, num_pruned_weights, num_pruned_weights / total_num_weights))
    print('Avg weight magnitude: {}'.format(weights.abs().mean()))


def make_weights_zero(net):
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = module.weight.data.abs().gt(1e-3).float()
            module.weight.data.mul_(mask)


def start_LSQ(net):
    pruner_mask_dict = {}
    def pruner_backward_hook(module_, grad_input, _):
        layer_name_ = get_layer_name(net, module_)
        mod_grad_input = tuple([grad.clone() if grad is not None else None
                                for grad in grad_input])
        if isinstance(module_, torch.nn.Conv2d):
            mod_grad_input[1].mul_(pruner_mask_dict[layer_name_])
        elif isinstance(module_, torch.nn.Linear):
            mod_grad_input[2].mul_(
                pruner_mask_dict[layer_name_].transpose(dim0=0, dim1=1))
        else:
            raise ValueError('Layer {}\'s type, i.e., {} is not supported!'.
                             format(layer_name_, module_))
        return mod_grad_input

    # actually prune the model
    make_weights_zero(net)
    count_sparsity(net)

    layer_names = get_all_layer_names(net, (nn.Conv2d, nn.Linear))
    for layer_name in layer_names:
        module = get_layer_by_name(net, layer_name)
        pruner_mask_dict[layer_name] = module.weight.data.abs().gt(0.0).float()
        module.register_backward_hook(pruner_backward_hook)


def get_constraint(bits, obj):
    if bits == 0:
        return None
    if 'activation' in obj:
        lower = 0
        upper = 2 ** bits
    elif 'swish' in obj:
        lower = -1
        upper = 2 ** bits - 1
    else:
        lower = -2 ** (bits - 1) + 1
        upper = 2 ** (bits - 1)
    constraint = np.arange(lower, upper)
    return constraint


def train_one_epoch(s_net, t_net, epoch, train_loader, optimizer, optimizer_t, criterion, criterion_kl_1step, criterion_kl_2step, log_fun=None):
    print('\n ... Training Model For One Epoch ...')
    s_net.train()
    t_net.train()
    msg_print = MessagePrinter(len(train_loader), freq=10)
    for index, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        optimizer_t.zero_grad()

        inputs, targets = inputs.cuda(), targets.long().cuda()

        outputs1 = s_net(inputs)
        outputs2 = t_net(inputs)

        loss_CE = criterion(outputs1, targets) + criterion(outputs2, targets)
        loss_KL = criterion_kl_1step(outputs1, outputs2) + criterion_kl_1step(outputs2, outputs1)

        loss = loss_CE + loss_KL

        loss.backward()
        # manager.pruner.non_uniform(True)

        optimizer.step()
        optimizer_t.step()

        # print msg and write log
        msg = msg_print(loss, targets, outputs1)
        if log_fun is not None:
            log_fun(index, s_net, msg)
    return msg_print.get_performance()


@autograd.no_grad()
def eval_performance(net, test_loader, criterion, log_fun=None):
    print('\n ... Evaluate Model Performance ...')
    net.eval()
    msg_print = MessagePrinter(len(test_loader), freq=10)
    for index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.long().cuda()
        outputs = net(inputs)
        loss_classify = criterion(outputs, targets)
        msg = msg_print(loss_classify, targets, outputs)
        if log_fun is not None:
            log_fun(index, net, msg)
    return msg_print.get_performance()

class KLLoss(nn.Module):
    def __init__(self):

        super(KLLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=2

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


class KLLoss_t3(nn.Module):
    def __init__(self):

        super(KLLoss_t3, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss


class KLLoss_lowtem(nn.Module):
    def __init__(self):

        super(KLLoss_lowtem, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=1

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

class MessagePrinter:
    def __init__(self, total_iter, freq=10):
        self.iter_total = total_iter
        self.iter_curr = 0
        self.loss_cls = 0
        self.num_total = 0
        self.num_top1_correct = 0
        self.num_top5_correct = 0
        self.freq = freq

    def get_performance(self):
        loss_cls = round(self.loss_cls, 4)
        top1_acc = round(100. * self.num_top1_correct / self.num_total, 4)
        top5_acc = round(100. * self.num_top5_correct / self.num_total, 4)
        return [loss_cls, top1_acc, top5_acc]

    def __call__(self, loss_cls, targets, outputs):
        loss_cls = loss_cls.item()
        self.loss_cls = (self.loss_cls * self.iter_curr + loss_cls) / (self.iter_curr + 1)
        self.iter_curr += 1

        top1_correct, top5_correct = self.top1_top5_correct(targets, outputs)
        self.num_total += targets.shape[0]
        self.num_top1_correct += top1_correct
        self.num_top5_correct += top5_correct

        time_stamp = str(datetime.now())[:19]
        msg = '{}, {}/{}, Cls={:.3f}, Top1={:.3f}, Top5={:.3f}'
        top1_acc = 100. * self.num_top1_correct / self.num_total
        top5_acc = 100. * self.num_top5_correct / self.num_total
        show_msg = msg.format(time_stamp,
                              self.iter_curr,
                              self.iter_total,
                              self.loss_cls,
                              top1_acc,
                             top5_acc)
        if self.iter_curr % self.freq == 0 or self.iter_curr == self.iter_total:
            print(show_msg)
        return show_msg

    @staticmethod
    @autograd.no_grad()
    def top1_top5_correct(target, output, topk=(1, 5)):
        maxk = max(topk)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).sum().item()
            res.append(correct_k)
        return res


class Trainer:
    def __init__(self, net, t_net, train_loader, test_loader, optimizer,optimizer_t, lr_scheduler,lr_scheduler_t, model_name, train_loger=None,
                 pruned=False):
        self.net = net
        self.t_net = t_net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.optimizer_t = optimizer_t
        self.pruned = pruned

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_t = lr_scheduler_t

        self.model_name = model_name
        self.train_loger = train_loger
        self.criterion = nn.CrossEntropyLoss()
        self.KLloss_1step = KLLoss()
        self.KLloss_2step = KLLoss_lowtem()

    def save_model(self):
        print('\n ... Save Model ...')
        model_path = os.path.join(self.train_loger.model_cache, self.model_name + '.pth')
        try:
            state_dict = self.net.module.state_dict()
        except AttributeError:
            state_dict = self.net.state_dict()
        torch.save(state_dict, model_path)

    def load_check_point(self):
        ckpt_cache = self.train_loger.ckpt_cache
        net_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_net.pth')
        opt_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_opt.pth')
        lrs_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_lrs.pth')
        self.net.load_state_dict(torch.load(net_path))
        self.optimizer.load_state_dict(torch.load(opt_path))
        self.lr_scheduler.load_state_dict(torch.load(lrs_path))

        epoch_acc_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_epoch_acc.pkl')
        with open(epoch_acc_path, 'rb') as fd:
            check_point = pickle.load(fd)
            start_epoch = check_point['start_epoch']
            best_quan_acc = check_point['best_quan_acc']
        return start_epoch, best_quan_acc

    def save_check_point(self, epoch, best_quan_acc):
        ckpt_cache = self.train_loger.ckpt_cache
        net_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_net.pth')
        opt_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_opt.pth')
        lrs_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_lrs.pth')
        torch.save(self.net.state_dict(), net_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        torch.save(self.lr_scheduler.state_dict(), lrs_path)

        epoch_acc_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_epoch_acc.pkl')
        with open(epoch_acc_path, 'wb') as fd:
            check_point = {'start_epoch': epoch + 1, 'best_quan_acc': best_quan_acc}
            pickle.dump(check_point, fd)

    def __call__(self, total_epoch, save_check_point=True, resume=False, ic_manager=None):
        start_epoch = 0
        best_quan_acc = 0
        if resume:
            start_epoch, best_quan_acc = self.load_check_point()
            print('\n Resume, start_epoch=%d, best_quan_acc=%.3f' % (start_epoch, best_quan_acc))

        for epoch in range(start_epoch, total_epoch):
            print("\n %s | Current: %d | Total: %d" % (datetime.now(), epoch + 1, total_epoch))
            if ic_manager:
                ic_manager(epoch)
            train_perf_epoch = train_one_epoch(s_net=self.net,
                                               t_net=self.t_net,
                                               epoch=epoch,
                                               train_loader=self.train_loader,
                                               optimizer=self.optimizer,
                                               optimizer_t=self.optimizer_t,
                                               criterion=self.criterion,
                                               criterion_kl_1step=self.KLloss_1step,
                                               criterion_kl_2step=self.KLloss_2step,
                                               log_fun=self.train_loger.print_log)
            if self.pruned:
                make_weights_zero(self.net)
            quan_perf_epoch = eval_performance(net=self.net,
                                               test_loader=self.test_loader,
                                               criterion=self.criterion,
                                               log_fun=self.train_loger.print_log)
            self.lr_scheduler.step()
            self.lr_scheduler_t.step()

            self.train_loger.print_perf(train_perf_epoch + quan_perf_epoch)
            if quan_perf_epoch[1] > best_quan_acc:
                best_quan_acc = quan_perf_epoch[1]
                self.save_model()

            if self.pruned:
                count_sparsity(self.net)

            if save_check_point:
                self.save_check_point(epoch, best_quan_acc)

class Trainer_t3:
    def __init__(self, net, t_net, train_loader, test_loader, optimizer,optimizer_t, lr_scheduler,lr_scheduler_t, model_name, train_loger=None,
                 pruned=False):
        self.net = net
        self.t_net = t_net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.optimizer_t = optimizer_t
        self.pruned = pruned

        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_t = lr_scheduler_t

        self.model_name = model_name
        self.train_loger = train_loger
        self.criterion = nn.CrossEntropyLoss()
        self.KLloss_1step = KLLoss_t3()
        self.KLloss_2step = KLLoss_lowtem()

    def save_model(self):
        print('\n ... Save Model ...')
        model_path = os.path.join(self.train_loger.model_cache, self.model_name + '.pth')
        try:
            state_dict = self.net.module.state_dict()
        except AttributeError:
            state_dict = self.net.state_dict()
        torch.save(state_dict, model_path)

    def load_check_point(self):
        ckpt_cache = self.train_loger.ckpt_cache
        net_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_net.pth')
        opt_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_opt.pth')
        lrs_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_lrs.pth')
        self.net.load_state_dict(torch.load(net_path))
        self.optimizer.load_state_dict(torch.load(opt_path))
        self.lr_scheduler.load_state_dict(torch.load(lrs_path))

        epoch_acc_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_epoch_acc.pkl')
        with open(epoch_acc_path, 'rb') as fd:
            check_point = pickle.load(fd)
            start_epoch = check_point['start_epoch']
            best_quan_acc = check_point['best_quan_acc']
        return start_epoch, best_quan_acc

    def save_check_point(self, epoch, best_quan_acc):
        ckpt_cache = self.train_loger.ckpt_cache
        net_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_net.pth')
        opt_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_opt.pth')
        lrs_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_lrs.pth')
        torch.save(self.net.state_dict(), net_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        torch.save(self.lr_scheduler.state_dict(), lrs_path)

        epoch_acc_path = os.path.join(ckpt_cache, self.model_name + '_ckpt_epoch_acc.pkl')
        with open(epoch_acc_path, 'wb') as fd:
            check_point = {'start_epoch': epoch + 1, 'best_quan_acc': best_quan_acc}
            pickle.dump(check_point, fd)

    def __call__(self, total_epoch, save_check_point=True, resume=False):
        start_epoch = 0
        best_quan_acc = 0
        if resume:
            start_epoch, best_quan_acc = self.load_check_point()
            print('\n Resume, start_epoch=%d, best_quan_acc=%.3f' % (start_epoch, best_quan_acc))

        for epoch in range(start_epoch, total_epoch):
            print("\n %s | Current: %d | Total: %d" % (datetime.now(), epoch + 1, total_epoch))
            train_perf_epoch = train_one_epoch(s_net=self.net,
                                               t_net=self.t_net,
                                               epoch=epoch,
                                               train_loader=self.train_loader,
                                               optimizer=self.optimizer,
                                               optimizer_t=self.optimizer_t,
                                               criterion=self.criterion,
                                               criterion_kl_1step=self.KLloss_1step,
                                               criterion_kl_2step=self.KLloss_2step,
                                               log_fun=self.train_loger.print_log)
            if self.pruned:
                make_weights_zero(self.net)
            quan_perf_epoch = eval_performance(net=self.net,
                                               test_loader=self.test_loader,
                                               criterion=self.criterion,
                                               log_fun=self.train_loger.print_log)
            self.lr_scheduler.step()
            self.lr_scheduler_t.step()

            self.train_loger.print_perf(train_perf_epoch + quan_perf_epoch)
            if quan_perf_epoch[1] > best_quan_acc:
                best_quan_acc = quan_perf_epoch[1]
                self.save_model()

            if self.pruned:
                count_sparsity(self.net)

            if save_check_point:
                self.save_check_point(epoch, best_quan_acc)

