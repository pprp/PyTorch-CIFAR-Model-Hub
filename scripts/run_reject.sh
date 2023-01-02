#!/bin/bash 

# factor 0
python tools/train.py --smooth_factor 0 --model lenet --dataset mnist --root ./data --crit lsr

# factor 0.1
python tools/train.py --smooth_factor 0.1 --model lenet --dataset mnist --root ./data --crit lsr

# factor 0.2
python tools/train.py --smooth_factor 0.2 --model lenet --dataset mnist --root ./data --crit lsr