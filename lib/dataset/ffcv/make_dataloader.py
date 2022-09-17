import time
from typing import List

import torch
import torchvision
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (Convert, Cutout, RandomHorizontalFlip,
                             RandomTranslate, ToDevice, ToTensor, ToTorchImage)
from ffcv.transforms.common import Squeeze


def make_dataloaders(train_dataset=None,
                     val_dataset=None,
                     batch_size=None,
                     num_workers=None):
    paths = {'train': train_dataset, 'test': val_dataset}

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice('cuda:0'),
            Squeeze()
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice('cuda:0', non_blocking=True),
            ToTorchImage(),
            Convert(torch.float32),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=batch_size,
                               num_workers=num_workers,
                               order=ordering,
                               drop_last=(name == 'train'),
                               pipelines={
                                   'image': image_pipeline,
                                   'label': label_pipeline
                               })

    return loaders, start_time
