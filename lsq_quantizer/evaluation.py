import argparse
import torch
from utils.mixnet import mixnet_s
from utils.data_loader import dataloader_imagenet
from helpers import load_checkpoint
from utils.utilities import get_constraint, eval_performance
from utils.add_lsqmodule import add_lsqmodule
from micronet_score import get_micronet_score

def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--data_root', default=None, type=str)

    parser.add_argument('--weight_bits', required=True,  type=int)
    parser.add_argument('--activation_bits', default=0, type=int)

    parser.add_argument('--batch_size', default=128, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    constr_activation = get_constraint(args.activation_bits, 'activation')
    net = mixnet_s(quan_first=True,
                  quan_last=True,
                  constr_activation=constr_activation,
                  preactivation=False,
                  bw_act=args.activation_bits)
    test_loader = dataloader_imagenet(args.data_root, split='test', batch_size=args.batch_size)
    add_lsqmodule(net, bit_width=args.weight_bits)

    name_weights_old = torch.load(args.model_path)
    name_weights_new = net.state_dict()
    name_weights_new.update(name_weights_old)
    load_checkpoint(net, name_weights_new)

    criterion = torch.nn.CrossEntropyLoss()

    score = get_micronet_score(net, args.weight_bits, args.activation_bits)
    
    # Calculate accuracy
    net = net.cuda()

    quan_perf_epoch = eval_performance(net, test_loader, criterion)
    accuracy = quan_perf_epoch[1]

    print("Accuracy:", accuracy)
    print("Score:", score)

main()
