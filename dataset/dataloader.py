from functools import total_ordering
from operator import pos
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.utils import RandomErase
from .autoaugment import CIFAR10Policy
from .cutout import Cutout


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
            post_transform.append(Cutout(1,16))

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


def build_dataset(type="train", name="cifar10", root="~/data", args=None, fast=False):
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
        ratio = 0.5
        total_num = len(dataset_type.targets)
        choice_num = int(total_num * ratio)
        print(f"Choice num/Total num: {choice_num}/{total_num}")
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
                build_dataset("train", "cifar100", args.root, args=args, fast=args.fast),
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
