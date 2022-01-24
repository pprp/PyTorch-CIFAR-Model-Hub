import argparse
from torch.cuda.amp import GradScaler as GradScaler


def str2bool(v):
    if v.lower() in ["true", 1]:
        return True
    elif v.lower() in ["false", 0]:
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="resnet20", help="select model")

    parser.add_argument(
        "--name", default="gpu0", help="experiment name: (default: cifar10_ricap)"
    )
    parser.add_argument("--gpu", default="0", type=str, help="gpu id")
    parser.add_argument(
        "--config",
        help="configuration file",
        type=str,
        default="lib/config/default.yaml",
    )

    ########### DATASET PART #####################
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="dataset name",
    )
    parser.add_argument("--root", default="~/datasets", help="root of dataset")
    parser.add_argument(
        "--fast", default=False, type=str2bool, help="train part of cifar10"
    )
    parser.add_argument("--bs", default=128, type=int, help="use RICAP")
    parser.add_argument("--nw", default=4, type=int, help="use RICAP")
    parser.add_argument("--half", default=False, type=bool, help="use AMP by hand")

    ########### CONTROLER ########################
    parser.add_argument("--geno", default="SPP1", help="genotype name")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", "--learning-rate", default=1e-1, type=float)
    parser.add_argument(
        "--optims",
        default="sgd",
        choices=["sgd", "adam", "adabound", "adaboundw", "nesterov", "sam", "asam", "adamw"],
        help="optimizer name support sgd, adabound",
    )
    parser.add_argument(
        "--sched",
        default="multistep",
        choices=["warmup", "multistep", "cosine", "warmcosine", "custom"],
        help="select scheduler",
    )
    parser.add_argument(
        "--crit",
        default="ce",
        choices=["ce", "lsr"],
        help="select criterion",
    )
    parser.add_argument("--milestones", default="60,120,160", type=str)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--nesterov", default=False, type=str2bool)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument(
        "--gradient_clip", default=2.0, type=float, help="gradient clip parameter"
    )
    ############# AUGMENTATION ####################
    # AUGMENTATION: ricap
    parser.add_argument("--ricap", default=False, type=str2bool, help="use RICAP")
    parser.add_argument("--ricap-beta", default=0.3, type=float)

    # AUGMENTATION: random erase
    parser.add_argument(
        "--random-erase", default=False, type=str2bool, help="use Random Erasing"
    )
    parser.add_argument("--random-erase-prob", default=0.5, type=float)
    parser.add_argument("--random-erase-sl", default=0.02, type=float)
    parser.add_argument("--random-erase-sh", default=0.4, type=float)
    parser.add_argument("--random-erase-r", default=0.3, type=float)

    # AUGMENTATION: autoaugmentation
    parser.add_argument(
        "--autoaugmentation", default=False, type=str2bool, help="use auto augmentation"
    )

    # AUGMENTATION: RandAugmentation

    # AUGMENTATION: cutout
    parser.add_argument("--cutout", default=False, type=str2bool, help="use cutout")

    # AUGMENTATION: mixup
    parser.add_argument("--mixup", default=False, type=str2bool, help="use Mixup")
    parser.add_argument("--mixup-alpha", default=1.0, type=float)

    # AUGMENTATION: cutmix
    parser.add_argument(
        "--cutmix", default=False, type=str2bool, help="using cutmix or not"
    )
    parser.add_argument("--cutmix_prob", default=0.5, type=float, help="cutmix prob")
    parser.add_argument("--beta", default=1.0, type=float, help="hyperparameter beta")

    args = parser.parse_args()

    if args.amp:
        args.scaler = GradScaler()

    return args
