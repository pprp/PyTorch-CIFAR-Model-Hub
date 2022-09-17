#!/bin/bash

python tools/retrain.py --model 'resnet20' --name "resnet20_rf" --sched 'cosine' --epochs 600 --fast False --dataset "cifar10" --geno SPP1

python tools/retrain.py --model 'resnet20' --name "resnet20_rf_ls" --sched 'cosine' --epochs 600 --fast False --dataset "cifar10" --geno SPP1 --crit "lsr"

# python tools/retrain.py --model 'resnet20' --name "resnet20_rf_cutout" --sched 'cosine' --epochs 600 --fast False --dataset "cifar10" --geno SPP1 --cutout True

# python tools/retrain.py --model 'resnet20' --name "resnet20_rf_cutout_ls" --sched 'cosine' --epochs 600 --fast False --dataset "cifar10" --geno SPP1 --cutout True  --crit "lsr"
