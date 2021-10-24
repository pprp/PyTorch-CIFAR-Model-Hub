python train.py --model 'shake_resnet26_2x64d' --name "base_shake" --sched 'cosine' --epochs 300
python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shakeC" --sched 'cosine' --epochs 300
python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shakeM" --sched 'cosine' --epochs 300
python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shakeMC" --sched 'cosine' --epochs 300
