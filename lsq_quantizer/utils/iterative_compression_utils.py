import torch
import numpy as np


def get_all_layer_names(model, subtypes=None):
    if subtypes is None:
        return [name for name, module in model.named_modules()][1:]
    return [name for name, module in model.named_modules() if isinstance(module, subtypes)]


def get_layer_name(model, layer):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


def get_layer_by_name(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, 'is_cuda'):
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().numpy()
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()

    return np.array(tensor)


class ICManager(object):
    """
    Manages various phases (i. e., warm-up and fine-tuning) within
    each iterative-compression cycle.

    Parameters:
        model:
            Model to be iteratively compressed
        pruning_method:
            Unstructured pruning method. Supported methods "L2" (Song Han's paper),
            and "Sensitivity-based" (EigenDamage paper)
        pruning_ratio:
            fraction of the parameters to be pruned away
        pruning_step:
            fraction of parameters to be pruned in each cycle
        total_epoch:
            Total length of training in epochs.
        """
    def __init__(self, model, pruning_method="L2", pruning_ratio=0.5,
                 pruning_step=0.05, total_epoch=20):
        assert 1 > pruning_ratio >= pruning_step > 0
        assert (total_epoch-1) > pruning_ratio / pruning_step
        self.model = model
        self.pruning_method = pruning_method
        self.pruning_ratio = pruning_ratio
        self.pruning_step = pruning_step
        self.curr_pruning = 0.
        self.cycle = int((total_epoch-1) * pruning_step / pruning_ratio)
        self.warm_up = 1
        self.layer_names = get_all_layer_names(model, torch.nn.Conv2d)
        self.layer_names.extend(get_all_layer_names(model, torch.nn.Linear))
        self.grad_logger_handle_dict = {}
        self.pruner_handle_list = []
        self.pruner_mask_dict = {}
        self.start_pruning()

    def __del__(self):
        self.end_pruning()

    def __call__(self, epoch):
        """
        :param epoch: gives length of training, in epochs, so far
        """

        # manage various compression phases
        if epoch < self.warm_up:
            return
        elif (epoch-self.warm_up) % self.cycle == 0:
            self.prune_further()
        else:
            return

    def start_pruning(self):
        def pruner_backward_hook(module_, grad_input, _):
            layer_name_ = get_layer_name(self.model, module_)
            mod_grad_input = tuple([grad.clone() if grad is not None else None
                                    for grad in grad_input])
            if isinstance(module_, torch.nn.Conv2d):
                mod_grad_input[1].mul_(self.pruner_mask_dict[layer_name_])
            elif isinstance(module_, torch.nn.Linear):
                """
                Apparently, for dense layers grad_input elements are switched:
                grad_input[0] is gradient tensor wrt filter biases of shape
                    [out_channels] if the bias exists.
                grad_input[1] is gradient tensor wrt input activations of shape
                    [batch_size, in_features]
                grad_input[2] is gradient tensor wrt kernels of shape
                    [in_features, out_channels] (Note that shape is transpose
                    of layer's weight matrix)
                It is not clear how this changes when bias doesn't exists.
                """
                mod_grad_input[2].mul_(
                    self.pruner_mask_dict[layer_name_].transpose(dim0=0, dim1=1))
            else:
                raise ValueError('Layer {}\'s type, i.e., {} is not supported!'.
                                 format(layer_name_, module_))
            return mod_grad_input

        assert len(self.pruner_handle_list) == 0
        for layer_name in self.layer_names:
            assert layer_name not in self.pruner_mask_dict
            module = get_layer_by_name(self.model, layer_name)
            self.pruner_mask_dict[layer_name] =\
                module.weight.data.clone().fill_(1).cuda()
            self.pruner_handle_list.append(module.register_backward_hook(
                pruner_backward_hook))

    def end_pruning(self):
        for handle in self.pruner_handle_list:
            handle.remove()
        self.pruner_handle_list = []

    def prune_further(self):
        if self.curr_pruning >= self.pruning_ratio:
            return
        self.curr_pruning += min(self.pruning_ratio - self.curr_pruning,
                                 self.pruning_step)
        if self.pruning_method == "L2":
            self.l2_pruning()
        else:
            raise ValueError('Unknown pruning method: {}'.format(
                self.pruning_method))

    def l2_pruning(self):
        total_num_weights = 0
        for layer_name in self.layer_names:
            module = get_layer_by_name(self.model, layer_name)
            total_num_weights += module.weight.data.numel()

        pruning_metric = torch.zeros(total_num_weights).cuda()
        idx = 0
        for layer_name in self.layer_names:
            module = get_layer_by_name(self.model, layer_name)
            num_weights = module.weight.data.numel()
            pruning_metric[idx:idx+num_weights] = \
                module.weight.data.view(-1).abs()
            idx += num_weights

        num_pruned_weights = total_num_weights -\
                             pruning_metric.abs().gt(1e-3).float().sum()
        print('\nSo far: # params: {}, pruned # params: {}, pruning ratio: {}'.
              format(total_num_weights, num_pruned_weights, num_pruned_weights /
                     total_num_weights))
        y, i = torch.sort(pruning_metric)
        thresh_idx = int(total_num_weights * self.curr_pruning)
        threshold = y[thresh_idx]

        num_pruned_weights = 0
        for layer_name in self.layer_names:
            module = get_layer_by_name(self.model, layer_name)
            assert layer_name in self.pruner_mask_dict
            mask = self.pruner_mask_dict[layer_name]
            mask.mul_(module.weight.data.abs().gt(threshold).float())
            module.weight.data.mul_(mask)
            num_pruned_weights += mask.numel() - torch.sum(mask)
            print('Layer: {} \t total # params: {:d} \t remaining params: {:d}'.
                  format(layer_name, mask.numel(), int(torch.sum(mask))))
        print('Total # params: {}, pruned # params: {}, pruning ratio: {}'.format(
            total_num_weights, num_pruned_weights, num_pruned_weights /
            total_num_weights))
