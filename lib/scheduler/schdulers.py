import itertools
from bisect import bisect_right

# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LambdaLR, OneCycleLR,
                                      ReduceLROnPlateau, StepLR, _LRScheduler)
from torch.optim.sgd import SGD


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self,
                 optimizer,
                 multiplier,
                 total_epoch,
                 after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr *
                ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch +
                 1.0) for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr *
                ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch +
                 1.0) for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# class StepLR:
#     def __init__(self, optimizer, learning_rate: float, total_epochs: int):
#         self.optimizer = optimizer
#         self.total_epochs = total_epochs
#         self.base = learning_rate

#     def __call__(self, epoch):
#         if epoch < self.total_epochs * 3 / 10:
#             lr = self.base
#         elif epoch < self.total_epochs * 6 / 10:
#             lr = self.base * 0.2
#         elif epoch < self.total_epochs * 8 / 10:
#             lr = self.base * 0.2**2
#         else:
#             lr = self.base * 0.2**3

#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = lr

#     def lr(self) -> float:
#         return self.optimizer.param_groups[0]['lr']


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


class CustomScheduler():
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate

    def __call__(self, epoch):
        if epoch < self.total_epochs * 3 / 10:
            lr = self.base
        elif epoch < self.total_epochs * 6 / 10:
            lr = self.base * 0.2
        elif epoch < self.total_epochs * 8 / 10:
            lr = self.base * 0.2**2
        else:
            lr = self.base * 0.2**3

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1 / 3,
        warmup_epochs=25,
        warmup_method='linear',
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of'
                ' increasing integers. Got {}',
                milestones,
            )

        if warmup_method not in ('constant', 'linear'):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                'got {}'.format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == 'constant':
                warmup_factor = self.warmup_factor
            elif self.warmup_method == 'linear':
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor *
            self.gamma**bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    initial_lr = 0.1
    total_epoch = 100
    net = model()

    optimizer = torch.optim.SGD(net.parameters(), lr=initial_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = StepLR(optimizer, initial_lr, total_epoch)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0.0001)
    # a_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                 lambda step: (1.0-step/total_epoch), last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                    milestones=[100, 150], last_epoch=-1,gamma=0.5)
    # scheduler = GradualWarmupScheduler(
    #     optimizer, 1, total_epoch=5, after_scheduler=a_scheduler
    # )
    # scheduler = WarmupMultiStepLR(optimizer, [100,150], gamma=0.5)
    # scheduler = CyclicLR(optimizer, base_lr=0, max_lr=0.1, step_size_up=15, step_size_down=20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=0.0001)
    # scheduler = LambdaLR(optimizer, lambda step : (1.0-step/total_epoch) if step > 15 else (step/total_epoch), last_epoch=-1)
    # scheduler = GradualWarmupScheduler(
    #     optimizer, 1, total_epoch=15, after_scheduler=a_scheduler
    # )
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    # scheduler = OneCycleLR(optimizer,)
    # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)

    print('初始化的学习率：', optimizer.defaults['lr'])

    lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化

    for epoch in range(0, total_epoch):
        optimizer.zero_grad()
        optimizer.step()
        print('第%d个epoch的学习率：%f' % (epoch, optimizer.param_groups[0]['lr']))
        print(scheduler.get_lr())
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # 画出lr的变化
    plt.plot(list(range(0, total_epoch)), lr_list)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title("learning rate's curve changes as epoch goes on!")
    # plt.show()
    plt.savefig('lr_show.png')
