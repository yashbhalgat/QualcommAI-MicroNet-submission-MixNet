import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lsq_module import Conv2d
from .lsq_module import Linear
from .lsq_module import LsqActivation
from .utilities import get_constraint

class _Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, constr_activation=None):
        super(BasicBlock, self).__init__()
        self.quan_activation = constr_activation is not None

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, constr_activation=None):
        super(NetworkBlock, self).__init__()
        self.constr_activation = constr_activation
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, constr_activation=self.constr_activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,
                 quan_first=False, quan_last=False, constr_activation=None, bw_act=None):
        super(WideResNet, self).__init__()

        self.quan_activation = constr_activation is not None
        self.quan_first = quan_first
        self.quan_last = quan_last

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        if self.quan_first:
            self.first_act = LsqActivation(get_constraint(bw_act, "weight")) if self.quan_activation else _Identity()
            self.conv1 = Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, constr_activation=constr_activation)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, constr_activation=constr_activation)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, constr_activation=constr_activation)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True) # Don't need to quantize here because this layer goes into AVG pooling

        if self.quan_last:
            self.last_act = LsqActivation(get_constraint(bw_act, "weight")) if self.quan_activation else _Identity()
            self.fc = Linear(nChannels[3], num_classes)
        else:
            self.fc = nn.Linear(nChannels[3], num_classes)

        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.quan_first:
            x = self.first_act(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)


        #out = out.view(-1, self.nChannels[3])
        out = out.view(-1, self.nChannels)
        if self.quan_last:
            out = self.last_act(out)

        return self.fc(out)


class WideResNet1(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet1, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)


        out = out.view(-1, self.nChannels[3])
        #out = out.view(-1, self.nChannels)

        return self.fc(out)



class WideResNet_custom(nn.Module):
    def __init__(self, num_blocks, base_channel, num_classes, dropRate=0.0):
        super(WideResNet_custom, self).__init__()
        nChannels = [16, base_channel, base_channel * 2, base_channel * 4]
        # assert((depth - 4) % 6 == 0)
        # n = (depth - 4) / 6
        n = num_blocks
        block = BasicBlock
        self.block = BasicBlock
        self.num_blocks = n
        self.num_classes = num_classes
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        return self.fc(out)

def WRN_custom(num_blocks, base_channel, num_classes = 10):
    return WideResNet_custom(num_blocks=num_blocks, base_channel=base_channel, num_classes=num_classes)

def WRN10_4(num_class = 10):
    return WideResNet(depth=10, num_classes=num_class, widen_factor=4)





def WRN16_1(num_class = 10):
    return WideResNet(depth=16, num_classes=num_class, widen_factor=1)




def WRN16_1_custom(num_class = 10):
    return WideResNet1(depth=16, num_classes=num_class, widen_factor=1)


def WRN16_4_custom(num_class = 10):
    return WideResNet1(depth=16, num_classes=num_class, widen_factor=4)


def WRN28_2_custom(num_class = 10):
    return WideResNet1(depth=28, num_classes=num_class, widen_factor=2)

def WRN16_2_custom(num_class = 10):
    return WideResNet1(depth=16, num_classes=num_class, widen_factor=2)









def WRN16_2(num_class = 10):
    return WideResNet(depth=16, num_classes=num_class, widen_factor=2)

def WRN16_4(num_class = 10):
    return WideResNet(depth=16, num_classes=num_class, widen_factor=4)

def WRN16_8(num_class = 10):
    return WideResNet(depth=16, num_classes=num_class, widen_factor=8)

def WRN28_2(num_class = 10):
    return WideResNet(depth=28, num_classes=num_class, widen_factor=2)

def WRN28_4(num_class = 10):
    return WideResNet(depth=28, num_classes=num_class, widen_factor=4)

def WRN22_4(num_class = 10):
    return WideResNet(depth=22, num_classes=num_class, widen_factor=4)

def WRN22_8(num_class = 10):
    return WideResNet(depth=22, num_classes=num_class, widen_factor=8)

def WRN28_1(num_class = 10):
    return WideResNet(depth=28, num_classes=num_class, widen_factor=1)

def WRN10_1(num_class = 10):
    return WideResNet(depth=10, num_classes=num_class, widen_factor=1)




def WRN28_4_custom(num_class = 10):
    return WideResNet1(depth=28, num_classes=num_class, widen_factor=4)



def WRN28_2_custom(num_class = 10):
    return WideResNet1(depth=28, num_classes=num_class, widen_factor=2)



def WRN40_1_custom(num_class = 10):
    return WideResNet1(depth=40, num_classes=num_class, widen_factor=1)



def WRN28_10(num_class = 10):
    return WideResNet(depth=28, num_classes=num_class, widen_factor=10)

def WRN28_10_custom(num_class = 10):
    return WideResNet1(depth=28, num_classes=num_class, widen_factor=10)



def WRN16_10(num_class = 10):
    return WideResNet(depth=16, num_classes=num_class, widen_factor=10)

def WRN16_10_custom(num_class = 10):
    return WideResNet1(depth=16, num_classes=num_class, widen_factor=10)



def WRN40_1(num_class = 10):
    return WideResNet(depth=40, num_classes=num_class, widen_factor=1)



def WRN34_4(num_class = 10):
    return WideResNet(depth=34, num_classes=num_class, widen_factor=4)

def WRN34_8(num_class = 10):
    return WideResNet(depth=34, num_classes=num_class, widen_factor=8)

def WRN34_5(num_class = 10):
    return WideResNet(depth=34, num_classes=num_class, widen_factor=5)

def WRN34_6(num_class = 10):
    return WideResNet(depth=34, num_classes=num_class, widen_factor=6)



def WRN40_4(quan_first=False, quan_last=False, constr_activation=None, preactivation=None, bw_act=None):
    return WideResNet(depth=40, num_classes=100, widen_factor=4,
                      quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation, bw_act=bw_act)

def WRN40_8(quan_first=False, quan_last=False, constr_activation=None, preactivation=None, bw_act=None):
    return WideResNet(depth=40, num_classes=100, widen_factor=8,
                      quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation, bw_act=bw_act)

def WRN40_5(quan_first=False, quan_last=False, constr_activation=None, preactivation=None, bw_act=None):
    return WideResNet(depth=40, num_classes=100, widen_factor=5,
                      quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation, bw_act=bw_act)

def WRN40_6(quan_first=False, quan_last=False, constr_activation=None, preactivation=None, bw_act=None):
    return WideResNet(depth=40, num_classes=100, widen_factor=6,
                      quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation, bw_act=bw_act)


def WRN46_4(num_class = 10):
    return WideResNet(depth=46, num_classes=num_class, widen_factor=4)

def WRN46_8(num_class = 10):
    return WideResNet(depth=46, num_classes=num_class, widen_factor=8)

def WRN46_5(num_class = 10):
    return WideResNet(depth=46, num_classes=num_class, widen_factor=5)

def WRN46_6(num_class = 10):
    return WideResNet(depth=46, num_classes=num_class, widen_factor=6)

