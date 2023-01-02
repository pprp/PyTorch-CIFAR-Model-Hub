echo on
E:
cd E:\GitHub\pytorch-cifar-tricks
python tools/train.py --smooth_factor 0 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.1 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.2 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.3 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.4 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.5 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.6 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.7 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.8 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 0.9 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32

python tools/train.py --smooth_factor 1 --model lenet --dataset mnist --root ./data --crit lsr --lr 0.01 --bs 32
