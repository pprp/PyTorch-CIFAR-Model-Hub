import logging
import os
import sys
import time

import _init_paths
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lib.core.function import validate
from lib.dataset import build_dataloader
from lib.models.build import build_model
from lib.utils.args import parse_args
from lib.utils.utils import *  # noqa: F401, F403


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)

    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if num_gpus > 1:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(
        os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000,
                                                  local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    val_loader = build_dataloader('cifar10', type='val')

    print('load data successfully')

    model = build_model()

    print('load model successfully')

    print('load from latest checkpoint')
    lastest_model = args.weights
    if lastest_model is not None:
        checkpoint = torch.load(lastest_model)
        model.load_state_dict(checkpoint['state_dict'])

    model = model.cuda(args.gpu)

    # 参数设置
    args.val_dataloader = val_loader

    print('start to validate model...')

    val_log = validate(args, val_loader, model, None, 0, writer=None)


if __name__ == '__main__':
    main()
