import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1,
                              c2,
                              k,
                              s,
                              autopad(k, p),
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (nn.SiLU() if act is True else
                    (act if isinstance(act, nn.Module) else nn.Identity()))
        # self.act = nn.ReLU(inplace=True)
        # self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SPP(nn.Module):
    # Spatial pyramid pooling
    def __init__(self, c1, c2, k=[3, 5, 7]):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
