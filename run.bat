echo on 
E:
cd E:\GitHub\pytorch-cifar-tricks
call python train.py --model "shake_resnet26_2x64d" --optims "sam" --name "shake_sam"
call python train.py --model "shake_resnet26_2x64d" --optims "asam" --name "shake_asam"