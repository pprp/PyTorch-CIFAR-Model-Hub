#!/bin/bash
# 10% datasets 

# module load anaconda/2021.05
# module load cuda/11.1
# module load cudnn/8.2.1_cuda11.x
# source activate hb
# export PYTHONUNBUFFERED=1

# python train.py --model 'spp_d11_pN' --name "spp_d11_pN" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" & \ 
# python train.py --model 'spp_d11_pA' --name "spp_d11_pA" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 

# python train.py --model 'spp_d11_pB' --name "spp_d11_pB" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" & \ 
# python train.py --model 'spp_d11_pC' --name "spp_d11_pC" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 

#python train.py --model 'spp_d11_pD' --name "spp_d11_pD" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 

####################################################################################################

# python train.py --model 'spp_d20_pN' --name "spp_d20_pN" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d20_pA' --name "spp_d20_pA" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d20_pB' --name "spp_d20_pB" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d20_pC' --name "spp_d20_pC" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d20_pD' --name "spp_d20_pD" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 

####################################################################################################

# python train.py --model 'spp_d29_pN' --name "spp_d29_pN" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d29_pA' --name "spp_d29_pA" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d29_pB' --name "spp_d29_pB" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
# python train.py --model 'spp_d29_pC' --name "spp_d29_pC" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
python train.py --model 'spp_d29_pD' --name "spp_d29_pD" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
