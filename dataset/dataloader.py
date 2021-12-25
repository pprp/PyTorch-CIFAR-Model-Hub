import random
from functools import lru_cache, total_ordering, partial
from operator import pos

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from utils.utils import RandomErase

from .autoaugment import CIFAR10Policy
from .cutout import Cutout
from utils.misc import *

# equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
# equals np.std(cifar10()['train']['data'], axis=(0,1,2))
cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87),
    (62.99, 62.09, 66.70),
]

custom_transformers = [
    partial(
        normalise,
        mean=np.array(cifar10_mean, dtype=np.float32),
        std=np.array(cifar10_std, dtype=np.float32),
    ),
    partial(transpose, source="NHWC", target="NCHW"),
]


@lru_cache(None)
def cifar10(root="~/data"):
    download = lambda train: datasets.CIFAR10(root=root, train=train, download=True)
    return {
        k: {"data": v.data, "targets": v.targets}
        for k, v in [
            ("train", download(train=True)),
            ("valid", download(train=False)),
        ]
    }


class CustomDataloader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        set_random_choices=False,
        num_workers=0,
        drop_last=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return (
            {"input": x.to(self.device).half(), "target": y.to(self.device).long()}
            for (x, y) in self.dataloader
        )# generater

    def __len__(self):
        return len(self.dataloader)


def build_transforms(name="cifar10", type="train", args=None):
    assert type in ["train", "val"]
    assert name in ["cifar10", "cifar100"]
    transform_type = None

    if type == "train":
        base_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if args.random_erase:
            mid_transform = [
                RandomErase(
                    args.random_erase_prob,
                    args.random_erase_sl,
                    args.random_erase_sh,
                    args.random_erase_r,
                ),
            ]
        elif args.autoaugmentation:
            mid_transform = [
                CIFAR10Policy(),
            ]
        else:
            mid_transform = []

        if name == "cifar10":
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        elif name == "cifar100":
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4865, 0.4409], [0.1942, 0.1918, 0.1958]
                ),
            ]

        if args.cutout:
            post_transform.append(Cutout(1, 8))

        transform_type = transforms.Compose(
            [*base_transform, *mid_transform, *post_transform]
        )

    elif type == "val":
        if name == "cifar10":
            transform_type = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        elif name == "cifar100":
            transform_type = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5071, 0.4865, 0.4409], [0.1942, 0.1918, 0.1958]
                    ),
                ]
            )
    else:
        raise "Type Error in transforms"
    return transform_type


def build_dataset(type="train", name="cifar10", root="~/data", args=None, fast=True):
    assert name in ["cifar10", "cifar100"]
    assert type in ["train", "val"]

    dataset_type = None

    if name == "cifar10":
        if type == "train":
            dataset_type = datasets.CIFAR10(
                root=root,
                train=True,
                download=True,
                transform=build_transforms("cifar10", "train", args=args),
            )
        elif type == "val":
            dataset_type = datasets.CIFAR10(
                root=root,
                train=False,
                download=True,
                transform=build_transforms("cifar10", "val", args=args),
            )

    elif name == "cifar100":
        if type == "train":
            dataset_type = datasets.CIFAR100(
                root=root,
                train=True,
                download=True,
                transform=build_transforms("cifar10", "train", args=args),
            )
        elif type == "val":
            dataset_type = datasets.CIFAR100(
                root=root,
                train=False,
                download=True,
                transform=build_transforms("cifar10", "val", args=args),
            )
    else:
        raise "Type Error: {} Not Supported".format(name)

    if fast:
        # fast train using ratio% images
        ratio = 0.3
        total_num = len(dataset_type.targets)
        choice_num = int(total_num * ratio)
        print(f"FAST MODE: Choice num/Total num: {choice_num}/{total_num}")

        dataset_type.data = dataset_type.data[:choice_num]
        dataset_type.targets = dataset_type.targets[:choice_num]

    print("DATASET:", len(dataset_type))

    return dataset_type


def build_dataloader(name="cifar10", type="train", args=None):
    assert type in ["train", "val"]
    assert name in ["cifar10", "cifar100"]

    dataloader_type = None
    num_classes = None
    if name == "cifar10":
        num_classes = 10
        if type == "train":
            dataloader_type = DataLoader(
                build_dataset("train", "cifar10", args.root, args=args, fast=args.fast),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
            )
        elif type == "val":
            dataloader_type = DataLoader(
                build_dataset("val", "cifar10", args.root, args=args, fast=args.fast),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
            )
    elif name == "cifar100":
        num_classes = 100
        if type == "train":
            dataloader_type = DataLoader(
                build_dataset(
                    "train", "cifar100", args.root, args=args, fast=args.fast
                ),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
            )
        elif type == "val":
            dataloader_type = DataLoader(
                build_dataset("val", "cifar100", args.root, args=args, fast=args.fast),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
            )
    else:
        raise "Type Error: {} Not Supported".format(name)

    return dataloader_type, num_classes
