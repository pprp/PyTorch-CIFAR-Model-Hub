#!/bin/bash

# python -m torch.utils.bottleneck train.py --model 'resnet20' --name "fast_training" --sched 'cosine' --epochs 1 --cutout True --sched "cosine" --lr 0.6 --bs 512 --nw 0 --fast True 


python train.py --model 'dawnnet' --name "fast_training" --sched 'custom' --epochs 35 --cutout False --lr 0.35 --bs 512 --nw 4

# gprof2dot -f pstats profile.prof | dot -Tpng -o out.png



# python train.py --model 'poolformer_s12' --name "poolformer_s12_dim128" --sched 'cosine' --epochs 200 --lr 0.01 

# python train.py --model 'vision_transformer' --name "vision_transformer" --sched 'cosine' --epochs 200 & \
# python train.py --model 'mobilevit_s' --name "mobilevit_s" --sched 'cosine' --epochs 200 

# python train.py --model 'mobilevit_xs' --name "mobilevit_xs" --sched 'cosine' --epochs 200 & \
# python train.py --model 'mobilevit_xxs' --name "mobilevit_xxs" --sched 'cosine' --epochs 200

# python train.py --model 'coatnet_0' --name "coatnet_0" --sched 'cosine' --epochs 200 --lr 0.01 & \
# python train.py --model 'coatnet_1' --name "coatnet_1" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'coatnet_2' --name "coatnet_2" --sched 'cosine' --epochs 200 --lr 0.01 & \
# python train.py --model 'coatnet_3' --name "coatnet_3" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'coatnet_4' --name "coatnet_4" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'cvt' --name "cvt" --sched 'cosine' --epochs 200 --lr 0.01 & \
# python train.py --model 'swin_t' --name "swin_t" --sched 'cosine' --epochs 200 --lr 0.01

# python train.py --model 'swin_s' --name "swin_s" --sched 'cosine' --epochs 200 --lr 0.01
# python train.py --model 'swin_b' --name "swin_b" --sched 'cosine' --epochs 200 --lr 0.01
# python train.py --model 'swin_l' --name "swin_l" --sched 'cosine' --epochs 200 --lr 0.01
# out of memory 

# python train.py --model 'shake_resnet26_2x64d' --name "base_shake" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shakeC" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shakeM" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shakeMC" --sched 'cosine' --epochs 200
