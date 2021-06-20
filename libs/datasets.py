import os
from abc import ABC

from randaugment import RandAugment
from torchvision import datasets
from torchvision.transforms import transforms


class DatasetGetter(ABC):
    def get(self, path):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError


class CIFARGetter(DatasetGetter, ABC):
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(32, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    @property
    def image_size(self):
        return 32

    @property
    def channels(self):
        return 3


class CIFAR10Getter(CIFARGetter):
    def get(self, path):
        train_ds = datasets.CIFAR10(root=path, train=True, download=True, transform=self.train_transform)
        val_ds = datasets.CIFAR10(root=path, train=False, download=True, transform=self.test_transform)
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 10


class CIFAR100Getter(CIFARGetter):
    def get(self, path):
        train_ds = datasets.CIFAR100(root=path, train=True, download=True, transform=self.train_transform)
        val_ds = datasets.CIFAR100(root=path, train=False, download=True, transform=self.test_transform)
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 100


class ImageNetGetter():
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def get(self, path):
        assert os.path.exists(path)
        train_ds = datasets.ImageNet(root=path, split='train', transform=self.train_transform)
        val_ds = datasets.ImageNet(root=path, split='val', transform=self.test_transform)
        return train_ds, val_ds

    @property
    def num_classes(self):
        return 1000

    @property
    def image_size(self):
        return 224

    @property
    def channels(self):
        return 3
