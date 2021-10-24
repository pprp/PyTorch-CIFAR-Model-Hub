# A PyTorch implementation of RICAP

This repository contains code for a data augmentation method **RICAP (Random Image Cropping And Patching)** based on [Data Augmentation using Random Image Cropping and Patching for Deep CNNs](https://arxiv.org/abs/1811.09030) implemented in PyTorch.

![example](example.png)

[TOC]

## Requirements

- Python 3.6
- PyTorch 0.4 or 1.0

## Tricks

- Warmup 
- Cosine LR Decay
- SAM
- Label Smooth
- KD
- Adabound
- Xavier Kaiming init



## Augmentation

- Auto Augmentation
- Cutout
- Mixup
- RICP
- Random Erase
- ShakeDrop



## Training

### CIFAR-10

WideResNet28-10 baseline on CIFAR-10:

```
python train.py --dataset cifar10
```

WideResNet28-10 +RICAP on CIFAR-10:

```
python train.py --dataset cifar10 --ricap True
```

WideResNet28-10 +Random Erasing on CIFAR-10:

```
python train.py --dataset cifar10 --random-erase True
```

WideResNet28-10 +Mixup on CIFAR-10:

```
python train.py --dataset cifar10 --mixup True
```

## Results

| Model                           |    Error rate     |   Loss    | Error rate (paper) |
| :------------------------------ | :---------------: | :-------: | :----------------: |
| WideResNet28-10 baseline        |   3.82（96.18）   |   0.158   |        3.89        |
| WideResNet28-10 +RICAP          | **2.82（97.18）** |   0.141   |      **2.85**      |
| WideResNet28-10 +Random Erasing |   3.18（96.82）   | **0.114** |        4.65        |
| WideResNet28-10 +Mixup          |   3.02（96.98）   |   0.158   |        3.02        |



reimplementation augmentation 

| Model                           |    Error rate     | Loss  | Error rate (paper) |
| :------------------------------ | :---------------: | :---: | :----------------: |
| WideResNet28-10 baseline        |   3.78（96.22）   |       |        3.89        |
| WideResNet28-10 +RICAP          | **2.81（97.19）** |       |      **2.85**      |
| WideResNet28-10 +Random Erasing |   3.03（96.97）   | 0.113 |        4.65        |
| WideResNet28-10 +Mixup          |   2.93（97.07）   | 0.158 |        3.02        |





reimplementation models(no augmentation, half data，epoch200，bs128)

| Model                        |  Error rate   |  Loss  |    Epoch(s)     | Params  |
| :--------------------------- | :-----------: | :----: | :-------------: | ------- |
| lenet(cpu爆炸)               |   （70.76）   |        |                 |         |
| wideresnet                   | 3.78（96.22） |        |                 |         |
| resnet20                     |   （89.72）   |        |                 |         |
| senet                        |   （92.34）   |        |                 |         |
| resnet18                     |   （92.08）   |        |                 |         |
| resnet34                     |   （92.48）   |        |                 |         |
| resnet50                     |   （91.72）   |        |                 |         |
| regnet                       |   （92.58）   |        |                 |         |
| nasnet                       |  out of mem   |        |                 |         |
| shake_resnet26_2x32d         |   （93.06）   |        |                 |         |
| shake_resnet26_2x64d         |   （94.14）   |        |                 |         |
| densenet                     |   （92.06）   |        |                 |         |
| dla                          |   （92.58）   |        |                 |         |
| googlenet                    |   （91.90）   | 0.2675 |                 |         |
| efficientnetb0(利用率低且慢) |   （86.82）   | 0.5024 |                 |         |
| mobilenet(利用率低)          |   （89.18）   |        |                 |         |
| mobilenetv2                  |   （91.06）   |        |                 |         |
| pnasnet                      |   （90.44）   |        |                 |         |
| preact_resnet                |   （90.76）   |        |                 |         |
| resnext                      |   （92.30）   |        |                 |         |
| vgg(cpugpu利用率都高)        |   （88.38）   |        |                 |         |
| inceptionv3                  |   （91.84）   |        |                 |         |
| inceptionv4                  |   （91.10）   |        |                 |         |
| inception_resnet_v2          |   （83.46）   |        |                 |         |
| rir                          |   （92.34）   | 0.3932 |                 |         |
| squeezenet(CPU利用率高)      |   （89.16）   | 0.4311 |       5s        |         |
| stochastic_depth_resnet18    |   （90.22）   |        |       6s        |         |
| xception                     |               |        |                 |         |
| dpn                          |   （92.06）   | 0.3002 |       24s       |         |
| ge_resnext29_8x64d           |   （93.86）   |  巨慢  | (**v100:gpu0**) | running |





TEST: scale/kernel ToyNet

结论：lenet这种卷积量比较少，只有两层的，cpu利用率高，gpu利用率低。在这个基础上增加深度，用vgg那种直筒方式增加深度，发现深度越深，cpu利用率越低，gpu利用率越高。

结论：bs会影响收敛效果。





stepLR 200 epoch

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
| shake_resnet26_2x64d | 200   |        |       | 96.33            |
| shake_resnet26_2x64d | 200   | √      |       | 96.99            |
| shake_resnet26_2x64d | 200   |        | √     | 96.60            |
| shake_resnet26_2x64d | 200   | √      | √     | 96.46            |







PS: `shake_resnet26_2x64d` achieved **97.71%** test accuracy with `cutout` and `mixup`!!

cosine lr

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
| shake_resnet26_2x64d | 300   |        |       |                  |
| shake_resnet26_2x64d | 300   | √      |       |                  |
| shake_resnet26_2x64d | 300   |        | √     |                  |
| shake_resnet26_2x64d | 300   | √      | √     |                  |



1800 epoch CIFAR ZOO中结果。

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
| shake_resnet26_2x64d | 1800  |        |       | 96.94            |
| shake_resnet26_2x64d | 1800  | √      |       | **97.20**        |
| shake_resnet26_2x64d | 1800  |        | √     | **97.42**        |
| shake_resnet26_2x64d | 1800  | √      | √     | **97.71**        |





## Divide and Co-training方案研究

- lr:
  - warmup (20 epoch)
  - cosine lr decay
  - lr=0.1
  - total epoch(300 epoch)
- bs=128
- aug:
  - Random Crop and resize
  - Random left-right flipping
  - AutoAugment
  - Normalization
  - Random Erasing
  - Mixup
- weight decay=5e-4 (bias and bn undecayed)
- kaiming weight init
- optimizer: nesterov



复现：((**v100:gpu1**)  4min*300/60=20h) top1: **97.59%**

```bash
python train.py --model 'pyramidnet272' \
                --name 'divide-co-train' \
                --autoaugmentation True \ 
                --random-erase True \
                --mixup True \
                --epochs 300 \
                --sched 'warmcosine' \
                --optims 'nesterov' \
                --bs 128 \
                --root '/home/dpj/project/data'
```



warmup (20 epoch)+ cosine +

| architecture | epoch | cutout | mixup | autoaugment | random-erase | C10 test acc (%) |
| ------------ | ----- | ------ | ----- | ----------- | ------------ | ---------------- |
| pyramid272   | 300   |        |       |             |              |                  |
| pyramid272   | 300   | √      |       |             |              |                  |
| pyramid272   | 300   |        | √     |             |              |                  |
| pyramid272   | 300   |        |       | √           |              |                  |
| pyramid272   | 300   |        |       |             | √            |                  |
| pyramid272   | 300   | √      | √     |             |              |                  |
| pyramid272   | 300   | √      |       | √           |              |                  |
| pyramid272   | 300   | √      |       |             | √            |                  |
| pyramid272   | 300   |        | √     | √           |              |                  |
| pyramid272   | 300   |        | √     |             | √            |                  |
| pyramid272   | 300   |        |       | √           | √            |                  |
|              |       |        |       |             |              |                  |
|              |       |        |       |             |              |                  |
|              |       |        |       |             |              |                  |
|              |       |        |       |             |              |                  |

```python
python train.py --model 'pyramid272' --name 'pyramid_orgin' 
python train.py --model 'pyramid272' --name 'pyramid_c' --cutout True
python train.py --model 'pyramid272' --name 'pyramid_m' --mixup True
python train.py --model 'pyramid272' --name 'pyramid_a' --autoaugmentation True 
python train.py --model 'pyramid272' --name 'pyramid_r' --random-erase True 
python train.py --model 'pyramid272' --name 'pyramid_cm'  --cutout True --mixup True
python train.py --model 'pyramid272' --name 'pyramid_ca' --cutout True --autoaugmentation True
python train.py --model 'pyramid272' --name 'pyramid_cr' --cutout True --random-erase True
python train.py --model 'pyramid272' --name 'pyramid_ma' --mixup True --autoaugmentation True
python train.py --model 'pyramid272' --name 'pyramid_mr' --mixup True --random-erase True
python train.py --model 'pyramid272' --name 'pyramid_ar' --autoaugmentation True --random-erase True 
```
