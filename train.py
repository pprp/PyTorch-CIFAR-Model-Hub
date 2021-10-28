import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.cuda.amp import autocast as autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataloader import build_dataloader
from model import build_model
from utils.controler import (
    build_criterion,
    build_expname,
    build_optimizer,
    build_scheduler,
)
from utils.utils import *
from utils.args import parse_args


def train(
    args, train_loader, model, criterion, optimizer, epoch, scheduler=None, writer=None
):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    train_tqdm = tqdm(train_loader, total=len(train_loader))
    train_tqdm.set_description(
        "[%s%03d/%03d %s%f]"
        % ("Epoch:", epoch + 1, args.epochs, "lr:", scheduler.get_last_lr()[0])
    )
    for i, (input, target) in enumerate(train_tqdm):
        # from original paper's appendix
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
        elif args.optims in ["sam", "asam"]:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.ascent_step()

            acc = accuracy(output, target)[0]
        else:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc = accuracy(output, target)[0]

        postfix = {"train_loss": "%.3f" % (loss.item()), "train_acc": "%.3f" % acc}

        train_tqdm.set_postfix(log=postfix)
        # compute gradient and do optimizing step
        if args.amp:
            optimizer.zero_grad()
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        elif args.optims in ["sam", "asam"]:
            loss = criterion(model(input), target)
            loss.backward()
            optimizer.descent_step()
        else:
            optimizer.zero_grad()
            loss.backward()
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
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

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
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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

    criterion = build_criterion(args).cuda()

    cudnn.benchmark = True

    # data loading code
    train_loader, num_classes = build_dataloader(args.dataset, type="train", args=args)
    test_loader, num_classes = build_dataloader(args.dataset, type="val", args=args)

    # create model
    model = build_model(args.model, num_classes)
    print(count_params(model))

    # stat(model, (3, 32, 32))
    # from torchsummary import summary
    # summary(model, input_size=(3, 32, 32), batch_size=-1)

    model = model.cuda()

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(args, optimizer)

    log = pd.DataFrame(
        index=[], columns=["epoch", "lr", "loss", "acc", "val_loss", "val_acc"]
    )

    best_acc = 0
    for epoch in range(args.epochs):
        print("Epoch [%d/%d]" % (epoch + 1, args.epochs))

        scheduler.step()

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
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion, epoch, writer=writer)

        print(
            "loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f"
            % (train_log["loss"], train_log["acc"], val_log["loss"], val_log["acc"])
        )

        tmp = pd.Series(
            [
                epoch,
                scheduler.get_lr()[0],
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
            print("=> saved best model")


if __name__ == "__main__":
    main()
