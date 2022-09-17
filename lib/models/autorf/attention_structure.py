import torch.nn as nn

from .operations import *
from .spaces import OPS


class LAAttention(nn.Module):
    def __init__(self, step, C, genotype):
        super(LAAttention, self).__init__()
        self._steps = step
        self._C = C
        self._ops = nn.ModuleList()
        self.C_in = self._C // 4
        self.C_out = self._C
        self.width = 4
        self.se = SE(self.C_in, reduction=2)  # 8
        self.se2 = SE(self.C_in * 4, reduction=2)  # 8
        self.channel_back = nn.Sequential(
            nn.Conv2d(self.C_in * 5,
                      self._C,
                      kernel_size=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(self._C),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._C,
                      self._C,
                      kernel_size=1,
                      padding=0,
                      groups=1,
                      bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.genotype = genotype
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](self.C_in, 1, True)
            self._ops += [op]
        self.indices = indices

    def forward(self, x):
        states = [x]
        C_num = x.shape[1]
        length = C_num // 4
        spx = torch.split(x, length, 1)
        spx_sum = sum(spx)
        spx_sum = self.se(spx_sum)
        states[0] = spx[0]
        h01 = states[self.indices[0]]
        op01 = self._ops[0]
        h01_out = op01(h01)
        s = h01_out
        states += [s]

        states[0] = spx[1]
        h02 = states[self.indices[1]]
        h12 = states[self.indices[2]]
        op02 = self._ops[1]
        op12 = self._ops[2]
        h02_out = op02(h02)
        h12_out = op12(h12)
        s = h02_out + h12_out
        states += [s]

        states[0] = spx[2]
        h03 = states[self.indices[3]]
        h13 = states[self.indices[4]]
        h23 = states[self.indices[5]]
        op03 = self._ops[3]
        op13 = self._ops[4]
        op23 = self._ops[5]
        h03_out = op03(h03)
        h13_out = op13(h13)
        h23_out = op23(h23)
        s = h03_out + h13_out + h23_out
        states += [s]

        states[0] = spx[3]
        h04 = states[self.indices[6]]
        h14 = states[self.indices[7]]
        h24 = states[self.indices[8]]
        h34 = states[self.indices[9]]

        op04 = self._ops[6]
        op14 = self._ops[7]
        op24 = self._ops[8]
        op34 = self._ops[9]

        h04_out = op04(h04)
        h14_out = op14(h14)
        h24_out = op24(h24)
        h34_out = op34(h34)
        s = h04_out + h14_out + h24_out + h34_out
        states += [s]

        node_concat = torch.cat(states[-4:], dim=1)
        node_concat = torch.cat((node_concat, spx_sum), dim=1)
        attention_out = self.channel_back(node_concat) + x
        attention_out = self.se2(attention_out)
        return attention_out


class ReceptiveFieldAttention(nn.Module):
    def __init__(self, C, steps=3, reduction=False, se=False, genotype=None):
        super(ReceptiveFieldAttention, self).__init__()
        assert genotype is not None
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.C_in = C

        self.genotype = genotype
        op_names, indices = zip(*self.genotype.normal)
        concat = genotype.normal_concat

        self.conv1x1 = nn.Conv2d(C * self._steps,
                                 C,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)

        if self._se:
            self.se = SE(self.C_in, reduction=4)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](self.C_in, 1, True)
            self._ops += [op]

        self.indices = indices

    def forward(self, x):
        states = [x]
        offset = 0

        total_step = (1 + self._steps) * self._steps // 2

        for i in range(total_step):
            h = states[self.indices[i]]
            ops = self._ops[i]
            s = ops(h)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)
        node_out = self.conv1x1(node_out)
        # shortcut
        node_out = node_out + x
        if self._se:
            node_out = self.se(node_out)

        return node_out
