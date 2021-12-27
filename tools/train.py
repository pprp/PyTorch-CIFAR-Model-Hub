from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml

import _init_paths

from dataset.loader import build_dataloader
from models import build_model
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.args import parse_args
from utils.controller import (build_criterion, build_expname, build_optimizer,
                              build_scheduler)
from utils.misc import Timer
from utils.utils import *


HALF_FLAG = False


def train(
    args, train_loader, model, criterion, optimizer, epoch, scheduler=None, writer=None
):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()
    if HALF_FLAG:
        model = model.half()

    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()

        if args.ricap:
            I_x, I_y = input.size()[2:]

            w = int(np.round(I_x * np.random.beta(args.ricap_beta, args.ricap_beta)))
            h = int(np.round(I_y * np.random.beta(args.ricap_beta, args.ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(input.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = input[idx][
                    :, :, x_k : x_k + w_[k], y_k : y_k + h_[k]
                ]
                c_[k] = target[idx].cuda()
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (
                    torch.cat((cropped_images[0], cropped_images[1]), 2),
                    torch.cat((cropped_images[2], cropped_images[3]), 2),
                ),
                3,
            )
            patched_images = patched_images.cuda()

            if args.amp:
                with autocast():
                    output = model(patched_images)
                    loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])
            else:
                output = model(patched_images)
                loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])

            acc = sum([W_[k] * accuracy(output, c_[k])[0] for k in range(4)])
        elif args.mixup:
            l = np.random.beta(args.mixup_alpha, args.mixup_alpha)

            idx = torch.randperm(input.size(0))
            input_a, input_b = input, input[idx]
            target_a, target_b = target, target[idx]

            mixed_input = l * input_a + (1 - l) * input_b

            target_a = target_a.cuda()
            target_b = target_b.cuda()
            mixed_input = mixed_input.cuda()

            output = model(mixed_input)
            loss = l * criterion(output, target_a) + (1 - l) * criterion(
                output, target_b
            )

            acc = (
                l * accuracy(output, target_a)[0]
                + (1 - l) * accuracy(output, target_b)[0]
            )
        elif args.cutmix:
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
            )
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
                1.0 - lam
            )

            acc = accuracy(output, target)[0]
        elif args.optims in ["sam", "asam"]:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.ascent_step()
            acc = accuracy(output, target)[0]
        else:
            # logging.info("train.py line 134")
            if HALF_FLAG:
                input = input.cuda().half()
            else:
                input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            acc, _ = accuracy(output, target, topk=(1, 5))

        # compute gradient and do optimizing step
        if args.amp:
            # optimizer.zero_grad()
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        elif args.optims in ["sam", "asam"]:
            loss = criterion(model(input), target)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.descent_step()
        else:
            # optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))

    log = OrderedDict(
        [
            ("loss", losses.avg),
            ("acc", scores.avg),
        ]
    )
    if writer is not None:
        writer.add_scalar("Train/Loss", losses.avg, epoch)
        writer.add_scalar("Train/Acc", scores.avg, epoch)

    return log


def validate(args, val_loader, model, criterion, epoch, writer):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if HALF_FLAG:
                input = input.cuda().half()
            else:
                input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, _ = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            scores.update(acc1.item(), input.size(0))

    log = OrderedDict(
        [
            ("loss", losses.avg),
            ("acc", scores.avg),
        ]
    )

    if writer is not None:
        writer.add_scalar("Val/Loss", losses.avg, epoch)
        writer.add_scalar("Val/Acc", scores.avg, epoch)

    return log


def main():
    cudnn.benchmark = True

    args = parse_args()

    # process argparse & yaml
    if not args.config:
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    else:  # yaml priority is higher than args
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(vars(args))
        args = argparse.Namespace(**opt)

    args.name = build_expname(args)

    writer = SummaryWriter(
        "exps/%s/runs/%s-%05d"
        % (args.name, time.strftime("%m-%d", time.localtime()), random.randint(0, 100))
    )

    if not os.path.exists("exps/%s" % args.name):
        os.makedirs("exps/%s" % args.name)

    print("--------Config -----")
    for arg in vars(args):
        print("%s: %s" % (arg, getattr(args, arg)))
    print("--------------------")

    with open("exps/%s/args.txt" % args.name, "w") as f:
        for arg in vars(args):
            print("%s: %s" % (arg, getattr(args, arg)), file=f)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=log_format,
        datefmt="%m/%d %I:%M:%S %p",
    )

    fh = logging.FileHandler(os.path.join("exps", args.name, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    #######################################################################

    train_loader = build_dataloader(args.dataset, type="train", args=args)
    test_loader = build_dataloader(args.dataset, type="val", args=args)

    # create model
    model = build_model(args.model, num_classes=10)
    logging.info(f"param of model {args.model} is {count_params(model)}")

    # stat(model, (3, 32, 32))
    # from torchsummary import summary
    # summary(model, input_size=(3, 32, 32), batch_size=-1)

    model = model.cuda()

    criterion = build_criterion(args).cuda()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(args, optimizer)

    log = pd.DataFrame(
        index=[], columns=["epoch", "lr", "loss", "acc", "val_loss", "val_acc"]
    )

    logging.info("Training Start...")

    timer = Timer()

    best_acc = 0
    for epoch in range(args.epochs):
        logging.info("Epoch [%03d/%03d]" % (epoch, args.epochs))
        # train for one epoch
        train_log = train(
            args,
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler=scheduler,
            writer=writer,
        )

        train_time = timer()

        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion, epoch, writer=writer)

        scheduler.step()

        logging.info(
            f"Epoch:[{epoch:03d}/{args.epochs:03d}] lr:{scheduler.get_last_lr()[0]:.4f} train_acc:{train_log['acc']/100.:.3f} train_loss:{train_log['loss']:.4f} train_time:{train_time:03.2f} valid_acc:{val_log['acc']/100.:.3f} valid_loss:{val_log['loss']:.4f} valid_time:{timer():03.2f} total_time: {timer()+train_time:.2f}"
        )

        tmp = pd.Series(
            [
                epoch,
                scheduler.get_last_lr()[0],
                train_log["loss"],
                train_log["acc"],
                val_log["loss"],
                val_log["acc"],
            ],
            index=["epoch", "lr", "loss", "acc", "val_loss", "val_acc"],
        )

        log = log.append(tmp, ignore_index=True)
        log.to_csv("exps/%s/log.csv" % args.name, index=False)

        if val_log["acc"] > best_acc:
            torch.save(
                model.state_dict(),
                "exps/%s/model_%d.pth" % (args.name, (val_log["acc"] * 100)),
            )
            best_acc = val_log["acc"]


if __name__ == "__main__":
    main()
