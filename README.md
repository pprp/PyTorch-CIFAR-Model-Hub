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

| Model                                 |  Error rate   | Loss | Epoch(s) | Params |
| :------------------------------------ | :-----------: | :--: | :------: | ------ |
| lenet                                 |   （70.76）   |      |          |        |
| wideresnet                            | 3.78（96.22） |      |          |        |
| resnet20                              |   （89.72）   |      |          |        |
| senet                                 |   （92.34）   |      |          |        |
| resnet18                              |   （92.08）   |      |          |        |
| resnet34                              |   （92.48）   |      |          |        |
| resnet50(**v100:gpu0**)               |    running    |      |          |        |
| regnet(220:0)                         |   （92.58）   |      |          |        |
| nasnet                                |  out of mem   |      |          |        |
| shake_resnet26_2x32d                  |   （93.06）   |      |          |        |
| shake_resnet26_2x64d（**v100:gpu1**） |    running    |      |          |        |
| densenet                              |   （92.06）   |      |          |        |
| dla                                   |               |      |          |        |
| googlenet                             |               |      |          |        |
| shufflenet                            |               |      |          |        |
| shufflenetv2                          |               |      |          |        |
| efficientnetb0                        |               |      |          |        |
| mobilenet                             |               |      |          |        |
| mobilenetv2                           |               |      |          |        |
| pnasnet                               |               |      |          |        |
| preact_resnet                         |               |      |          |        |
| resnext                               |               |      |          |        |
| vgg                                   |               |      |          |        |
| attention56                           |               |      |          |        |
| attention92                           |               |      |          |        |
| inceptionv3                           |               |      |          |        |
| inceptionv4                           |               |      |          |        |
| inception_resnet_v2                   |               |      |          |        |
| rir                                   |               |      |          |        |
| squeezenet                            |               |      |          |        |
| stochastic_depth_resnet18             |               |      |          |        |
| xception                              |               |      |          |        |
| dpn                                   |               |      |          |        |
| ge_resnext29_8x64d                    |               |      |          |        |
| ge_resnext29_16x64d                   |               |      |          |        |
| sk_resnext29_16x32d                   |               |      |          |        |
| sk_resnext29_16x64d                   |               |      |          |        |
| cbam_resnext29_16x64d                 |               |      |          |        |
| cbam_resnext29_8x64d                  |               |      |          |        |





reimplementation scale/kernel ToyNet

| Model   | Error rate | Loss | CPU(%) | GPU(%) |
| :------ | :--------: | :--: | :----: | ------ |
| s=1,k=5 |            |      |        |        |
| s=2,k=5 |            |      |        |        |
| s=3,k=5 |            |      |        |        |
| s=4,k=5 |            |      |        |        |
| s=2,k=3 |            |      |        |        |
|         |            |      |        |        |

