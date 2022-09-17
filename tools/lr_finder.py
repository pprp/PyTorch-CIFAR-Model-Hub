import argparse

import _init_paths
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim

from lib.dataset import build_dataloader
from lib.models import build_model
from lib.utils.utils import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_lr',
                        type=float,
                        default=1e-7,
                        help='min learning rate')
    parser.add_argument('--max_lr',
                        type=float,
                        default=10,
                        help='max learning rate')
    parser.add_argument('--num_iter',
                        type=int,
                        default=100,
                        help='num of iteration')
    parser.add_argument('--gpus',
                        nargs='+',
                        type=int,
                        default=0,
                        help='gpu device')
    parser.add_argument('--model', default='wideresnet', help='select model')

    parser.add_argument('--name',
                        default=None,
                        help='model name: (default: cifar10_ricap)')
    parser.add_argument(
        '--dataset',
        default='cifar10',
        choices=['cifar10', 'cifar100'],
        help='dataset name',
    )
    parser.add_argument('--bs', default=128, type=int, help='use RICAP')
    parser.add_argument('--nw', default=4, type=int, help='use RICAP')

    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='60,120,160', type=str)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--amp', type=str2bool, default=False)
    ######################################################################
    # AUGMENTATION: ricap
    parser.add_argument('--ricap',
                        default=False,
                        type=str2bool,
                        help='use RICAP')
    parser.add_argument('--ricap-beta', default=0.3, type=float)

    # AUGMENTATION: random erase
    parser.add_argument('--random-erase',
                        default=False,
                        type=str2bool,
                        help='use Random Erasing')
    parser.add_argument('--random-erase-prob', default=0.5, type=float)
    parser.add_argument('--random-erase-sl', default=0.02, type=float)
    parser.add_argument('--random-erase-sh', default=0.4, type=float)
    parser.add_argument('--random-erase-r', default=0.3, type=float)

    # AUGMENTATION: autoaugmentation
    parser.add_argument('--autoaugmentation',
                        default=False,
                        type=str2bool,
                        help='use auto augmentation')

    # AUGMENTATION: cutout
    parser.add_argument('--cutout',
                        default=False,
                        type=str2bool,
                        help='use cutout')

    # AUGMENTATION: mixup
    parser.add_argument('--mixup',
                        default=False,
                        type=str2bool,
                        help='use Mixup')
    parser.add_argument('--mixup-alpha', default=1.0, type=float)
    #####################################################################
    args = parser.parse_args()

    print(args)

    train_dataloader, num_classes = build_dataloader('cifar10',
                                                     'train',
                                                     args=args)

    net = build_model(args.model, num_classes)

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]

    net = nn.DataParallel(net, device_ids=args.gpus)
    net = net.cuda()

    lsr_loss = nn.CrossEntropyLoss().cuda()

    # apply no weight decay on bias
    params = split_weights(net)

    optimizer = optim.SGD(params,
                          lr=args.base_lr,
                          momentum=0.9,
                          weight_decay=1e-4,
                          nesterov=True)

    # set up warmup phase learning rate scheduler
    lr_scheduler = FindLR(optimizer,
                          max_lr=args.max_lr,
                          num_iter=args.num_iter)
    epoches = int(args.num_iter / len(train_dataloader)) + 1

    n = 0
    learning_rate = []
    losses = []
    for epoch in range(epoches):

        # training procedure
        net.train()

        for batch_index, (images, labels) in enumerate(train_dataloader):
            if n > args.num_iter:
                break

            lr_scheduler.step()

            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            predicts = net(images)
            loss = lsr_loss(predicts, labels)
            if torch.isnan(loss).any():
                n += 1e8
                break
            loss.backward()
            optimizer.step()

            n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1
            print(
                'Iterations: {iter_num} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f}'
                .format(
                    loss.item(),
                    optimizer.param_groups[0]['lr'],
                    iter_num=n,
                    trained_samples=batch_index * args.bs + len(images),
                    total_samples=len(train_dataloader.dataset),
                ))

            learning_rate.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())
            n += 1

    learning_rate = learning_rate[10:-5]
    losses = losses[10:-5]

    fig, ax = plt.subplots(1, 1)
    ax.plot(learning_rate, losses)
    ax.set_xlabel('learning rate')
    ax.set_ylabel('losses')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))

    fig.savefig('result.jpg')
