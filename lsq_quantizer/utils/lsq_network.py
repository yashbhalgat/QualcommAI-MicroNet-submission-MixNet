import math
import torch
import torch.nn as nn
from .lsq_module import Conv2d
from .lsq_module import Linear
from .lsq_module import LsqActivation


def _make_layer(block, in_channels, planes, nblocks, stride=1, constr_activation=None):
    layers = list()
    downsample = stride != 1 or in_channels != planes * block.expansion
    layers.append(block(in_channels, planes, stride, downsample, constr_activation))
    in_channels = planes * block.expansion
    for i in range(1, nblocks):
        layers.append(block(in_channels, planes, constr_activation=constr_activation))
    return nn.Sequential(*layers), planes * block.expansion


class _Identity(nn.Module):
    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, planes, stride=1, downsample=False, constr_activation=None):
        super(BasicBlock, self).__init__()
        self.quan_activation = constr_activation is not None

        self.conv1 = Conv2d(in_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.activation1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.activation2 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            conv = Conv2d(in_channels, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            bn = nn.BatchNorm2d(num_features=planes)
            self.downsample = nn.Sequential(*[conv, bn])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activation2(out)
        return out


class PreActivationBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, planes, stride=1, downsample=False, constr_activation=None):
        super(PreActivationBlock, self).__init__()
        self.quan_activation = constr_activation is not None

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.activation1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_channels, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.activation2 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.downsample = None
        if downsample:
            bn = nn.BatchNorm2d(num_features=in_channels)
            activation = LsqActivation(constr_activation) if self.quan_activation else _Identity()
            conv = Conv2d(in_channels, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.downsample = nn.Sequential(*[bn, activation, conv])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.bn1(x)
        out = self.activation1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation2(out)
        out = self.conv2(out)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, stride=1, downsample=None, constr_activation=None):
        super(Bottleneck, self).__init__()
        self.quan_activation = constr_activation is not None

        self.conv1 = Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.activation2 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activation3 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)

        self.downsample = None
        if downsample:
            conv = Conv2d(in_channels, planes * 4, kernel_size=1, stride=stride, padding=0, bias=False)
            bn = nn.BatchNorm2d(num_features=planes * 4)
            self.downsample = nn.Sequential(*[conv, bn])

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.activation3(out)
        return out


class Resnet20(nn.Module):
    def __init__(self, block, quan_first=False, quan_last=False, constr_activation=None):
        super(Resnet20, self).__init__()
        self.quan_first = quan_first
        self.quan_last = quan_last
        self.quan_activation = constr_activation is not None

        if quan_first:
            self.first_act = LsqActivation(constr_activation) if self.quan_activation else _Identity()
            self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.activation1 = LsqActivation(constr_activation) if self.quan_activation else nn.ReLU(inplace=True)
        in_channels = 16
        self.layer1, in_channels = _make_layer(block, in_channels, planes=16, nblocks=3,
                                               stride=1, constr_activation=constr_activation)
        self.layer2, in_channels = _make_layer(block, in_channels, planes=32, nblocks=3,
                                               stride=2, constr_activation=constr_activation)
        self.layer3, in_channels = _make_layer(block, in_channels, planes=64, nblocks=3,
                                               stride=2, constr_activation=constr_activation)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        if quan_last:
            self.last_act = LsqActivation(constr_activation) if self.quan_activation else _Identity()
            self.fc = Linear(in_features=64, out_features=100, bias=True)
        else:
            self.fc = nn.Linear(in_features=64, out_features=100, bias=True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.quan_first:
            x = self.first_act(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        if self.quan_last:
            out = self.last_act(out)
        out = self.fc(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, quan_first=False, quan_last=False, constr_activation=None):
        super(ResNet, self).__init__()
        self.quan_first = quan_first
        self.quan_last = quan_last
        self.quan_activation = constr_activation is not None
        self.constr_activation = constr_activation

        if self.quan_first:
            self.first_act = LsqActivation(constr_activation) if self.quan_activation else _Identity() 
            self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if self.quan_activation:
            self.activation1 = LsqActivation(constr_activation)
        else:
            self.activation1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = 64
        self.layer1, in_channels = _make_layer(block, in_channels, planes=64, nblocks=layers[0],
                                               stride=1, constr_activation=constr_activation)
        self.layer2, in_channels = _make_layer(block, in_channels, planes=128, nblocks=layers[1],
                                               stride=2, constr_activation=constr_activation)
        self.layer3, in_channels = _make_layer(block, in_channels, planes=256, nblocks=layers[2],
                                               stride=2, constr_activation=constr_activation)
        self.layer4, in_channels = _make_layer(block, in_channels, planes=512, nblocks=layers[3],
                                               stride=2, constr_activation=constr_activation)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if self.quan_last:
            self.last_act = LsqActivation(constr_activation) if self.quan_activation else _Identity()
            self.fc = Linear(512 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.quan_first:
            x = self.first_act(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.quan_last:
            x = self.last_act(x)
        x = self.fc(x)
        return x


def resnet18(quan_first=False, quan_last=False, constr_activation=None, preactivation=False):
    block = PreActivationBlock if preactivation else BasicBlock
    model = ResNet(block, [2, 2, 2, 2], quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation)
    return model


def resnet20(quan_first=False, quan_last=False, constr_activation=None, preactivation=False):
    block = PreActivationBlock if preactivation else BasicBlock
    model = Resnet20(block, quan_first, quan_last, constr_activation)
    return model


def resnet50(quan_first=False, quan_last=False, constr_activation=None, preactivation=False):
    block = Bottleneck
    model = ResNet(block, [3, 4, 6, 3], quan_first=quan_first, quan_last=quan_last, constr_activation=constr_activation)
    return model
