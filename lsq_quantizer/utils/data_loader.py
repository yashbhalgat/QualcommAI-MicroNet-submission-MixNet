import os
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def dataloader_cifar10(data_root, split='train', batch_size=128):
    if split == 'train':
        data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = True
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = False

    dataset = datasets.CIFAR10(data_root, train=train_flag, transform=data_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def dataloader_cifar100(data_root, split='train', batch_size=128):
    if split == 'train':
        data_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = True
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_flag = False

    dataset = datasets.CIFAR100("/prj/neo_lv/user/ybhalgat/LSQ-KD/", train=train_flag, transform=data_transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return loader


def dataloader_imagenet(data_root, split, batch_size):
    if split == 'train':
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data_root = os.path.join(data_root, 'train')
    else:
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data_root = os.path.join(data_root, 'val')

    dataset = torchvision.datasets.ImageFolder(data_root, transform=data_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return loader





