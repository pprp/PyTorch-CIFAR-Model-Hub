import logging
import os
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from lib.core.evaluate import accuracy

from lib.core.function import validate
from torchvision.datasets import mnist
from lib.models.build import build_model
from lib.utils.args import parse_args
from lib.utils.utils import *  # noqa: F401, F403
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if num_gpus > 1:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(
        os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000,
                                                  local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    train_dataset = mnist.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./data', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=5)
    test_loader = DataLoader(test_dataset, batch_size=5)
    val_loader = test_loader 

    print('load data successfully')

    model = build_model(args.model)

    print('load model successfully')

    print('load from latest checkpoint')
    # lastest_model = args.weights
    # if lastest_model is not None:
    #     checkpoint = torch.load(lastest_model)
    #     model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    # 参数设置
    args.val_dataloader = val_loader

    print('start to validate model...')

    criterion = nn.CrossEntropyLoss()

    # metric = accuracy()
    momentum = 0.9
    learning_rate = 0.001

    # minimize the loss function to network prediction, 
    # instead of the network parameter w.
    for i, (images, labels) in enumerate(train_loader):
        
        for _ in range(500):
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()

            # update with learning rate of 0.001 and momumtum of 0.9
            output = output * momentum - learning_rate * model.gradient     



if __name__ == '__main__':
    main()
