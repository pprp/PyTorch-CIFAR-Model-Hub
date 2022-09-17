echo on
E:
cd E:\GitHub\pytorch-cifar-tricks
@REM call python train.py --model "shake_resnet26_2x64d" --optims "sam" --name "shake_sam"
@REM call python train.py --model "shake_resnet26_2x64d" --optims "asam" --name "shake_asam"
call python train.py --model "shake_resnet26_2x64d" --sched "cosine" --name "cosine_decay"
call python train.py --model "shake_resnet26_2x64d" --crit "lsr" --name "label_smooth"
