python train.py --model 'shake_resnet26_2x64d' --name "original_shake"
python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shake_cutout"
python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shake_mixup"
python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shake_mixup_cutout"
