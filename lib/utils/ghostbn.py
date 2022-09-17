import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.BatchNorm2d):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 weight=True,
                 bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean',
                             torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var',
                             torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (
                mode is
                False):  #lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(
                self.num_splits, self.num_features),
                                           dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(
                self.num_splits, self.num_features),
                                          dim=0).repeat(self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(input.view(-1, C * self.num_splits, H, W),
                                self.running_mean, self.running_var,
                                self.weight.repeat(self.num_splits),
                                self.bias.repeat(self.num_splits), True,
                                self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(input, self.running_mean[:self.num_features],
                                self.running_var[:self.num_features],
                                self.weight, self.bias, False, self.momentum,
                                self.eps)
