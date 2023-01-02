import logging
import os
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from lib.core.loss import LSR

from lib.core.evaluate import accuracy
from lib.core.function import validate
from lib.models.build import build_model
from lib.utils.args import parse_args
from lib.utils.utils import *  # noqa: F401, F403
from lib.utils.rank import kendalltau, pearson, spearman, rank_difference, concordant_pair_ratio


def compute_autoloss_zero(criterion, model, train_loader):
    momentum = 0.9
    learning_rate = 0.001

    metric_list = []
    loss_list = []
    output_list = []

    # minimize the loss function to network prediction,
    # instead of the network parameter w.
    for i, (images, labels) in enumerate(train_loader):
        # get the network prediction optimized with loss function L
        output = model(images)
        for _ in range(500):
            loss = criterion(output, labels)
            loss.backward(retain_graph=True)

            # update with learning rate of 0.001 and momumtum of 0.9
            output = output * momentum - learning_rate * model.gradient
        output_list.append(output)
        loss_list.append(criterion(output, labels))
        metric_list.append(accuracy(output, labels))

        if i > 5:
            break

    # compute the correlation score
    corr_score_list = []
    for i, (images, labels) in enumerate(train_loader):
        output = model(images)
        corr_score = accuracy(output_list[i], labels)[0] - accuracy(output, labels)[0]
        if isinstance(corr_score, torch.Tensor):
            corr_score = corr_score.item()
        corr_score_list.append(corr_score)

        if i > 5:
            break
    

    print("mean_corr_score: ", np.mean(corr_score_list))
    return np.mean(corr_score_list)


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if num_gpus > 1:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    # Log
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%d %I:%M:%S"
    )
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists("./log"):
        os.mkdir("./log")
    fh = logging.FileHandler(
        os.path.join(
            "log/train-{}{:02}{}".format(
                local_time.tm_year % 2000, local_time.tm_mon, t
            )
        )
    )
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    train_dataset = mnist.MNIST(
        root="./data", train=True, transform=ToTensor(), download=True
    )
    test_dataset = mnist.MNIST(
        root="./data", train=False, transform=ToTensor(), download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    val_loader = test_loader

    print("load data successfully")

    model = build_model(args.model)

    print("load model successfully")

    print("load from latest checkpoint")
    # lastest_model = args.weights
    # if lastest_model is not None:
    #     checkpoint = torch.load(lastest_model)
    #     model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()

    # 参数设置
    args.val_dataloader = val_loader

    print("start to validate model...")

    smooth_factor = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    pred_list = []
    gt_list = [97.33, 97.88, 97.63, 97.36, 97.38, 97.56, 98, 97.60, 96.97, 95.43]
    
    # rank consistency 
    for smooth in smooth_factor:
        criterion = LSR(smooth)
        az = compute_autoloss_zero(criterion, model, train_loader)
        pred_list.append(az)
    
    kd, sp, pr = kendalltau(gt_list, pred_list), spearman(gt_list, pred_list), pearson(gt_list, pred_list)
    print("kendalltau: ", kd)
    print("spearman: ", sp)
    print("pearson: ", pr)


if __name__ == "__main__":
    main()
