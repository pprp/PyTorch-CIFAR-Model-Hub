import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from .transforms import build_transforms


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
    if name == "cifar10":
        if type == "train":
            dataloader_type = DataLoader(
                build_dataset("train", "cifar10", args.root, args=args, fast=args.fast),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
                pin_memory=True,
            )
        elif type == "val":
            dataloader_type = DataLoader(
                build_dataset("val", "cifar10", args.root, args=args, fast=args.fast),
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.nw,
                pin_memory=True,
            )
    elif name == "cifar100":
        if type == "train":
            dataloader_type = DataLoader(
                build_dataset(
                    "train", "cifar100", args.root, args=args, fast=args.fast
                ),
                batch_size=args.bs,
                shuffle=True,
                num_workers=args.nw,
                pin_memory=True,
            )
        elif type == "val":
            dataloader_type = DataLoader(
                build_dataset("val", "cifar100", args.root, args=args, fast=args.fast),
                batch_size=args.bs,
                shuffle=False,
                num_workers=args.nw,
                pin_memory=True,
            )
    else:
        raise "Type Error: {} Not Supported".format(name)

    return dataloader_type
