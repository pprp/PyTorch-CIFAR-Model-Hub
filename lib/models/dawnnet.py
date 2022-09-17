from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class BatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        weight_freeze=False,
        bias_freeze=False,
        weight_init=1.0,
        bias_init=0.0,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            self.weight.data.fill_(weight_init)
        if bias_init is not None:
            self.bias.data.fill_(bias_init)
        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

batch_norm = partial(BatchNorm, weight_init=None, bias_init=None)


class Add(namedtuple('Add', [])):
    def __call__(self, x, y):
        return x + y


class Identity(namedtuple('Identity', [])):
    def __call__(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class Concat(nn.Module):
    def forward(self, *xs):
        return torch.cat(xs, 1)


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


def res_block(c_in, c_out, stride, **kw):
    block = {
        'bn1': batch_norm(c_in, **kw),
        'relu1': nn.ReLU(True),
        'branch': {
            'conv1':
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            'bn2':
            batch_norm(c_out, **kw),
            'relu2':
            nn.ReLU(True),
            'conv2':
            nn.Conv2d(c_out,
                      c_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
        },
    }
    projection = (stride != 1) or (c_in != c_out)
    if projection:
        block['conv3'] = (
            nn.Conv2d(c_in,
                      c_out,
                      kernel_size=1,
                      stride=stride,
                      padding=0,
                      bias=False),
            ['relu1'],
        )
    block['add'] = (Add(), [('conv3' if projection else 'relu1'),
                            'branch/conv2'])
    return block


def dawn_cfg(c=64,
             block=res_block,
             prep_bn_relu=False,
             concat_pool=True,
             **kw):
    if isinstance(c, int):
        c = [c, 2 * c, 4 * c, 4 * c]

    classifier_pool = ({
        'in': Identity(),
        'maxpool': nn.MaxPool2d(4),
        'avgpool': (nn.AvgPool2d(4), ['in']),
        'concat': (Concat(), ['maxpool', 'avgpool']),
    } if concat_pool else {
        'pool': nn.MaxPool2d(4)
    })

    return {
        'input': (None, []),
        'prep':
        union(
            {
                'conv':
                nn.Conv2d(
                    3, c[0], kernel_size=3, stride=1, padding=1, bias=False)
            },
            {
                'bn': batch_norm(c[0], **kw),
                'relu': nn.ReLU(True)
            } if prep_bn_relu else {},
        ),
        'layer1': {
            'block0': block(c[0], c[0], 1, **kw),
            'block1': block(c[0], c[0], 1, **kw),
        },
        'layer2': {
            'block0': block(c[0], c[1], 2, **kw),
            'block1': block(c[1], c[1], 1, **kw),
        },
        'layer3': {
            'block0': block(c[1], c[2], 2, **kw),
            'block1': block(c[2], c[2], 1, **kw),
        },
        'layer4': {
            'block0': block(c[2], c[3], 2, **kw),
            'block1': block(c[3], c[3], 1, **kw),
        },
        'final':
        union(
            classifier_pool,
            {
                'flatten':
                Flatten(),
                'linear':
                nn.Linear(2 * c[3] if concat_pool else c[3], 10, bias=True),
            },
        ),
        'logits':
        Identity(),
    }


def conv_bn(c_in, c_out, bn_weight_init=1.0, **kw):
    return {
        'conv':
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
        'bn':
        batch_norm(c_out, bn_weight_init=bn_weight_init, **kw),
        'relu':
        nn.ReLU(True),
    }


def basic_net(channels, weight, pool, **kw):
    return {
        'input': (None, []),
        'prep':
        conv_bn(3, channels['prep'], **kw),
        'layer1':
        dict(conv_bn(channels['prep'], channels['layer1'], **kw), pool=pool),
        'layer2':
        dict(conv_bn(channels['layer1'], channels['layer2'], **kw), pool=pool),
        'layer3':
        dict(conv_bn(channels['layer2'], channels['layer3'], **kw), pool=pool),
        'pool':
        nn.MaxPool2d(4),
        'flatten':
        Flatten(),
        'linear':
        nn.Linear(channels['layer3'], 10, bias=False),
        'logits':
        Mul(weight),
    }


def net(channels=None,
        weight=0.125,
        pool=nn.MaxPool2d(2),
        extra_layers=(),
        res_layers=('layer1', 'layer3'),
        **kw):
    channels = channels or {
        'prep': 64,
        'layer1': 128,
        'layer2': 256,
        'layer3': 512
    }

    residual = lambda c, **kw: {
        'in': Identity(),
        'res1': conv_bn(c, c, **kw),
        'res2': conv_bn(c, c, **kw),
        'add': (Add(), ['in', 'res2/relu']),
    }
    n = basic_net(channels, weight, pool, **kw)

    for layer in res_layers:
        n[layer]['residual'] = residual(channels[layer], **kw)

    for layer in extra_layers:
        n[layer]['extra'] = conv_bn(channels[layer], channels[layer], **kw)

    return n


class Network(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items():
            setattr(self, path.replace('/', '_'), val)

    def nodes(self):
        return (node for node, _ in self.graph.values())

    def forward(self, inputs):
        outputs = dict(inputs)
        # outputs = inputs
        for k, (node, ins) in self.graph.items():
            # only compute nodes that are not supplied as inputs.
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])
        return outputs

    def half(self):
        for node in self.nodes():
            if isinstance(node,
                          nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self


has_inputs = lambda node: type(node) is tuple


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


def normpath(path):
    # simplified os.path.normpath
    parts = []
    for p in path.split('/'):
        if p == '..':
            parts.pop()
        elif p.startswith('/'):
            parts = [p]
        else:
            parts.append(p)
    return '/'.join(parts)


def pipeline(net):
    return [('/'.join(path), (node if has_inputs(node) else (node, [-1])))
            for (path, node) in path_iter(net)]


def build_graph(net):
    flattened = pipeline(net)
    resolve_input = (
        lambda rel_path, path, idx: normpath('/'.join((path, '..', rel_path)))
        if isinstance(rel_path, str) else flattened[idx + rel_path][0])
    return {
        path:
        (node[0], [resolve_input(rel_path, path, idx) for rel_path in node[1]])
        for idx, (path, node) in enumerate(flattened)
    }


def DAWNNet(num_classes=10):
    cfg = dawn_cfg()
    return Network(cfg)


def _weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4), 'constant', 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes,
                              self.expansion * planes,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        # avg pool 72.63
        # max pool None
        # concate avg max:
        self.maxpool = nn.MaxPool2d(4)
        # self.avgpool = nn.AvgPool2d(4)
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.avgpool = nn.AdaptiveAvgPool2d(1) # 69.15
        self.linear = nn.Linear(256, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, out.size()[3])
        out = self.maxpool(out)
        # out = torch.cat([self.maxpool(out), self.avgpool(out)], 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def half(self):
        for node in self.children():
            node.half()
        return self


def resnet_dawn(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == '__main__':
    m = DAWNNet()
    print(m)
    # a = torch.zeros(3, 3, 32, 32)
    # b = torch.zeros(3)
    # inputs = {'data': a, 'targets':b}
    # print(m(inputs).shape)
