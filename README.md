# A PyTorch implementation of RICAP
This repository contains code for a data augmentation method **RICAP (Random Image Cropping And Patching)** based on [Data Augmentation using Random Image Cropping and Patching for Deep CNNs](https://arxiv.org/abs/1811.09030) implemented in PyTorch.

![example](example.png)

## Requirements
- Python 3.6
- PyTorch 0.4 or 1.0

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

| Model                        |  Error rate   |      Loss       | Epoch(s) | Params |
| :--------------------------- | :-----------: | :-------------: | :------: | ------ |
| lenet(cpu爆炸)               |   （70.76）   |                 |          |        |
| wideresnet                   | 3.78（96.22） |                 |          |        |
| resnet20                     |   （89.72）   |                 |          |        |
| senet                        |   （92.34）   |                 |          |        |
| resnet18                     |   （92.08）   |                 |          |        |
| resnet34                     |   （92.48）   |                 |          |        |
| resnet50                     |   （91.72）   |                 |          |        |
| regnet                       |   （92.58）   |                 |          |        |
| nasnet                       |  out of mem   |                 |          |        |
| shake_resnet26_2x32d         |   （93.06）   |                 |          |        |
| shake_resnet26_2x64d         |   （94.14）   |                 |          |        |
| densenet                     |   （92.06）   |                 |          |        |
| dla                          |   （92.58）   |                 |          |        |
| googlenet                    |   （91.90）   |     0.2675      |          |        |
| shufflenet                   |       x       |                 |          |        |
| shufflenetv2                 |       x       |                 |          |        |
| efficientnetb0(利用率低且慢) |   （86.82）   |     0.5024      |          |        |
| mobilenet(利用率低)          |   （89.18）   |                 |          |        |
| mobilenetv2                  |               |                 |          |        |
| pnasnet                      |               |                 |          |        |
| preact_resnet                |               |                 |          |        |
| resnext                      |               |                 |          |        |
| vgg(cpugpu利用率都高)        |   （88.38）   |                 |          |        |
| attention56                  |               |                 |          |        |
| attention92                  |      nan      |                 |   51s    |        |
| inceptionv3                  |               |                 |          |        |
| inceptionv4                  |               |                 |          |        |
| inception_resnet_v2          |               |                 |          |        |
| rir                          |   （92.34）   |     0.3932      |          |        |
| squeezenet(CPU利用率高)      |   （89.16）   |     0.4311      |    5s    |        |
| stochastic_depth_resnet18    |   （90.22）   | (**v100:gpu1**) |    6s    |        |
| xception                     |               |                 |          |        |
| dpn                          |               | (**v100:gpu0**) |          |        |
| ge_resnext29_8x64d           |               |      巨慢       |          |        |
| ge_resnext29_16x64d          |               |                 |          |        |
| sk_resnext29_16x32d          |               |       OOM       |          |        |
| sk_resnext29_16x64d          |               |       OOM       |          |        |
| cbam_resnext29_16x64d        |               |                 |          |        |
| cbam_resnext29_8x64d         |               |                 |          |        |





TEST: scale/kernel ToyNet

结论：lenet这种卷积量比较少，只有两层的，cpu利用率高，gpu利用率低。在这个基础上增加深度，用vgg那种直筒方式增加深度，发现深度越深，cpu利用率越低，gpu利用率越高。

结论：bs会影响收敛效果。





and the `√` means which additional method be used. 🍰

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
|                      |       |        |       |                  |
| shake_resnet26_2x64d | 1800  |        |       | 96.94            |
| shake_resnet26_2x64d | 1800  | √      |       | **97.20**        |
| shake_resnet26_2x64d | 1800  |        | √     | **97.42**        |
| shake_resnet26_2x64d | 1800  | √      | √     | **97.71**        |

PS: `shake_resnet26_2x64d` achieved **97.71%** test accuracy with `cutout` and `mixup`!!



