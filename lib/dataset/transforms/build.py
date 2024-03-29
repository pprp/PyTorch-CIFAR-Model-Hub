import torchvision.transforms as transforms

from .autoaugment import CIFAR10Policy
from .cutout import Cutout
from .randomerase import RandomErase


def build_transforms(name='cifar10', type='train', args=None):
    assert type in ['train', 'val']
    assert name in ['cifar10', 'cifar100', 'mnist']
    transform_type = None

    if type == 'train':
        if name == 'cifar10' or name == 'cifar100':
            base_transform = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        elif name == 'mnist':
            base_transform = [
                transforms.RandomCrop(28, padding=4),
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

        if name == 'cifar10':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ]
        elif name == 'cifar100':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409],
                                     [0.1942, 0.1918, 0.1958]),
            ]
        elif name == 'mnist':
            post_transform = [
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, )),
            ]

        if args.cutout:
            post_transform.append(Cutout(1, 8))

        transform_type = transforms.Compose(
            [*base_transform, *mid_transform, *post_transform])

    elif type == 'val':
        if name == 'cifar10':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
            ])
        elif name == 'cifar100':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409],
                                     [0.1942, 0.1918, 0.1958]),
            ])
        elif name == 'mnist':
            transform_type = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, )),
            ])
    else:
        raise 'Type Error in transforms'

    return transform_type
