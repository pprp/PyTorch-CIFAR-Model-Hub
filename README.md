# A PyTorch Implementation of CIFAR Tricks

![](figures/logo.png)

CIFAR10数据集上CNN模型、Transformer模型以及Tricks，数据增强，正则化方法等，并进行了实现。欢迎提issue或者进行PR。

## 0. Requirements

- Python 3.6+
- torch=1.8.0+cu111
- torchvision+0.9.0+cu111
- tqdm=4.26.0
- PyYAML=6.0
- einops
- torchsummary


## 1. Implements

### 1.0 Models

vision Transformer:

| Model              | GPU Mem | Top1:train | Top1:val | weight:M |
| ------------------ | ------- | ---------- | -------- | -------- |
| vision_transformer | 2869M   | 68.96      | 69.02    | 47.6     |
| mobilevit_s        | 2009M   | 98.83      | 92.50    | 19.2     |
| mobilevit_xs       | 1681M   | 98.22      | 91.77    | 7.78     |
| mobilevit_xxs      | 1175M   | 96.40      | 90.17    | 4.0      |
| coatnet_0          | 1433M   | 99.94      | 90.15    | 64.9     |
| coatnet_1          | 2089M   | 99.97      | 90.09    | 123      |
| coatnet_2          | 2405M   | 99.99      | 90.86    | 208      |
| cvt                | 2593M   | 94.64      | 84.74    | 75       |
| swin_t             | 3927M   | 93.24      | 86.09    | 104      |
| swin_s             | 6707M   | 90.27      | 83.68    | 184      |



### 1.1 Tricks

- Warmup
- Cosine LR Decay
- SAM
- Label Smooth
- KD
- Adabound
- Xavier Kaiming init
- lr finder

### 1.2 Augmentation

- Auto Augmentation
- Cutout
- Mixup
- RICAP
- Random Erase
- ShakeDrop



## 2. Training

### 2.1 CIFAR-10训练示例

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

更多脚本可以参考 scripts/run.sh

## 3. Results

### 3.1 原pytorch-ricap的结果

| Model                           |    Error rate     |   Loss    | Error rate (paper) |
| :------------------------------ | :---------------: | :-------: | :----------------: |
| WideResNet28-10 baseline        |   3.82（96.18）   |   0.158   |        3.89        |
| WideResNet28-10 +RICAP          | **2.82（97.18）** |   0.141   |      **2.85**      |
| WideResNet28-10 +Random Erasing |   3.18（96.82）   | **0.114** |        4.65        |
| WideResNet28-10 +Mixup          |   3.02（96.98）   |   0.158   |        3.02        |

### 3.2 Reimplementation结果

| Model                           |    Error rate     | Loss  | Error rate (paper) |
| :------------------------------ | :---------------: | :---: | :----------------: |
| WideResNet28-10 baseline        |   3.78（96.22）   |       |        3.89        |
| WideResNet28-10 +RICAP          | **2.81（97.19）** |       |      **2.85**      |
| WideResNet28-10 +Random Erasing |   3.03（96.97）   | 0.113 |        4.65        |
| WideResNet28-10 +Mixup          |   2.93（97.07）   | 0.158 |        3.02        |

### 3.3 Half data快速训练验证各网络结构

reimplementation models(no augmentation, half data，epoch200，bs128)

| Model                        |  Error rate   |  Loss  |
| :--------------------------- | :-----------: | :----: |
| lenet(cpu爆炸)               |   （70.76）   |        |
| wideresnet                   | 3.78（96.22） |        |
| resnet20                     |   （89.72）   |        |
| senet                        |   （92.34）   |        |
| resnet18                     |   （92.08）   |        |
| resnet34                     |   （92.48）   |        |
| resnet50                     |   （91.72）   |        |
| regnet                       |   （92.58）   |        |
| nasnet                       |  out of mem   |        |
| shake_resnet26_2x32d         |   （93.06）   |        |
| shake_resnet26_2x64d         |   （94.14）   |        |
| densenet                     |   （92.06）   |        |
| dla                          |   （92.58）   |        |
| googlenet                    |   （91.90）   | 0.2675 |
| efficientnetb0(利用率低且慢) |   （86.82）   | 0.5024 |
| mobilenet(利用率低)          |   （89.18）   |        |
| mobilenetv2                  |   （91.06）   |        |
| pnasnet                      |   （90.44）   |        |
| preact_resnet                |   （90.76）   |        |
| resnext                      |   （92.30）   |        |
| vgg(cpugpu利用率都高)        |   （88.38）   |        |
| inceptionv3                  |   （91.84）   |        |
| inceptionv4                  |   （91.10）   |        |
| inception_resnet_v2          |   （83.46）   |        |
| rir                          |   （92.34）   | 0.3932 |
| squeezenet(CPU利用率高)      |   （89.16）   | 0.4311 |
| stochastic_depth_resnet18    |   （90.22）   |        |
| xception                     |               |        |
| dpn                          |   （92.06）   | 0.3002 |
| ge_resnext29_8x64d           |   （93.86）   |  巨慢  |

### 3.4 测试cpu gpu影响

TEST: scale/kernel ToyNet

**修改网络的卷积层深度，并进行训练，可以得到以下结论：**

结论：lenet这种卷积量比较少，只有两层的，cpu利用率高，gpu利用率低。在这个基础上增加深度，用vgg那种直筒方式增加深度，发现深度越深，cpu利用率越低，gpu利用率越高。

**修改训练过程的batch size，可以得到以下结论：**

结论：bs会影响收敛效果。

### 3.5 StepLR优化下测试cutout和mixup

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
| shake_resnet26_2x64d | 200   |        |       | 96.33            |
| shake_resnet26_2x64d | 200   | √      |       | 96.99            |
| shake_resnet26_2x64d | 200   |        | √     | 96.60            |
| shake_resnet26_2x64d | 200   | √      | √     | 96.46            |

### 3.6 测试SAM,ASAM,Cosine,LabelSmooth

| architecture         | epoch | SAM  | ASAM | Cosine LR Decay | LabelSmooth | C10 test acc (%) |
| -------------------- | ----- | ---- | ---- | --------------- | ----------- | ---------------- |
| shake_resnet26_2x64d | 200   | √    |      |                 |             | 96.51            |
| shake_resnet26_2x64d | 200   |      | √    |                 |             | 96.80            |
| shake_resnet26_2x64d | 200   |      |      | √               |             | 96.61            |
| shake_resnet26_2x64d | 200   |      |      |                 | √           | 96.57            |

PS:其他库在加长训练过程（epoch=1800）情况下可以实现 `shake_resnet26_2x64d` achieved **97.71%** test accuracy with `cutout` and `mixup`!!

### 3.7 测试cosine lr + shake

| architecture         | epoch | cutout | mixup | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ---------------- |
| shake_resnet26_2x64d | 300   |        |       | 96.66            |
| shake_resnet26_2x64d | 300   | √      |       | **97.21**        |
| shake_resnet26_2x64d | 300   |        | √     | 96.90            |
| shake_resnet26_2x64d | 300   | √      | √     | 96.73            |

1800 epoch CIFAR ZOO中结果，由于耗时过久，未进行复现。

| architecture         | epoch | cutout | mixup | C10 test acc (%)       |
| -------------------- | ----- | ------ | ----- | ---------------------- |
| shake_resnet26_2x64d | 1800  |        |       | 96.94（cifar zoo）     |
| shake_resnet26_2x64d | 1800  | √      |       | **97.20**（cifar zoo） |
| shake_resnet26_2x64d | 1800  |        | √     | **97.42**（cifar zoo） |
| shake_resnet26_2x64d | 1800  | √      | √     | **97.71**（cifar zoo） |

### 3.8 Divide and Co-training方案研究

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

复现：((**v100:gpu1**)  4min*300/60=20h) top1: **97.59%** 本项目目前最高值。

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

### 3.9 测试多种数据增强

| architecture         | epoch | cutout | mixup | autoaugment | random-erase | C10 test acc (%) |
| -------------------- | ----- | ------ | ----- | ----------- | ------------ | ---------------- |
| shake_resnet26_2x64d | 200   |        |       |             |              | 96.42            |
| shake_resnet26_2x64d | 200   | √      |       |             |              | **96.49**        |
| shake_resnet26_2x64d | 200   |        | √     |             |              | 96.17            |
| shake_resnet26_2x64d | 200   |        |       | √           |              | 96.25            |
| shake_resnet26_2x64d | 200   |        |       |             | √            | 96.20            |
| shake_resnet26_2x64d | 200   | √      | √     |             |              | 95.82            |
| shake_resnet26_2x64d | 200   | √      |       | √           |              | 96.02            |
| shake_resnet26_2x64d | 200   | √      |       |             | √            | 96.00            |
| shake_resnet26_2x64d | 200   |        | √     | √           |              | 95.83            |
| shake_resnet26_2x64d | 200   |        | √     |             | √            | 95.89            |
| shake_resnet26_2x64d | 200   |        |       | √           | √            | 96.25            |

```python
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_orgin' --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_c' --cutout True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_m' --mixup True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_a' --autoaugmentation True  --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_r' --random-erase True  --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_cm'  --cutout True --mixup True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_ca' --cutout True --autoaugmentation True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_cr' --cutout True --random-erase True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_ma' --mixup True --autoaugmentation True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_mr' --mixup True --random-erase True --bs 64
python train.py --model 'shake_resnet26_2x64d' --name 'ss64_ar' --autoaugmentation True --random-erase True  --bs 64
```

### 3.10 测试注意力机制

| Model      | Top1:train | Top1:val | weight:M |
| ---------- | ---------- | -------- | -------- |
| spp_d11_pN | 100        | 86.79    | 7.36     |
| spp_d11_pA | 100        | 85.83    | 7.36     |
| spp_d11_pB | 100        | 85.66    | 7.36     |
| spp_d11_pC | 100        | 85.56    | 7.36     |
| spp_d11_pD | 100        | 85.73    | 7.36     |
| spp_d20_pN | 100        | 90.59    | 13.4     |
| spp_d20_pA | 100        | 89.96    | 13.4     |
| spp_d20_pB | 100        | 89.26    | 13.4     |
| spp_d20_pC | 100        | 89.69    | 13.4     |
| spp_d20_pD | 100        | 89.93    | 13.4     |
| spp_d29_pN | 99.99      | 89.56    | 19.4     |
| spp_d29_pA | 100        | 90.13    | 19.4     |
| spp_d29_pB | 100        | 90.16    | 19.4     |
| spp_d29_pC | 100        | 90.09    | 19.4     |
| spp_d29_pD | 100        | 90.06    | 19.4     |



## 4. Reference

[1] https://github.com/BIGBALLON/CIFAR-ZOO

[2] https://github.com/pprp/MutableNAS

[3] https://github.com/clovaai/CutMix-PyTorch

[4] https://github.com/4uiiurz1/pytorch-ricap

[5] https://github.com/NUDTNASLab/pytorch-image-models

[6] https://github.com/facebookresearch/LaMCTS

[7] https://github.com/Alibaba-MIIL/ImageNet21K

[8] https://myrtle.ai/learn/how-to-train-your-resnet/
