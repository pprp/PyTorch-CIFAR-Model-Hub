from __future__ import print_function

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from lib.utils.Smodules import *


class RFBFeature(nn.Module):
    def __init__(self):
        super(RFBFeature, self).__init__()
        layers = [
            nn.Conv2d(3, 64, kernel_size=3),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            # nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1),
            # nn.ReLU(), # 8
            nn.Conv2d(128, 256, kernel_size=3),
            # nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            # nn.ReLU(), # 13
        ]
        self.stem = nn.Sequential(*layers)
        # self.Norm = SPP(256,256)
        # self.Norm = ASPP(256, 256)
        self.Norm = DCN(256, 256)
        # self.Norm = BasicRFB(256, 256, scale = 1.0, visual=2)
        # self.Norm = nn.Conv2d(256, 256, 1, 1, 0)

    def forward(self, x):
        print(x.shape)
        x = self.stem(x)
        print('Before Norm: ', x.shape)
        if self.Norm:
            return self.Norm(x)
        return x


NAME = 'ASPP'

net = RFBFeature()  #build the net


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


def weight_init_random(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.normal_(m.state_dict()[key])
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


# net.Norm.apply(weights_init) #initial
net.apply(weight_init_random)
input_shape = [32, 32, 3]

imgt = Image.open(r'./dd.jpg', mode='r')
imgt = imgt.resize((input_shape[0], input_shape[1]), Image.ANTIALIAS)
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(32),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

x = trans(imgt)

x = x.unsqueeze(0)

# imgt = tile_pil_image(imgt, tile_factor=0, shade=True)
x = torch.randn(1, 3, 32, 32)
x = Variable(x, requires_grad=True)  #input
out = net(x)  #output

Zero_grad = torch.Tensor(1, 256, 10, 10).zero_()  #Zero_grad

Zero_grad[0][128][4][4] = 1  #set the middle pixel to 1.0

out.backward(Zero_grad)  #backward

z = x.grad.data.cpu().numpy()  #get input graident

print(z)
z = np.sum(np.abs(z), axis=1)

# z = np.mean(z,axis=1) #calculate mean by channels
# # z = np.array(z).mean(0).
# z /= z.max()
# z += (np.abs(z) > 0) * 0.2

#convert to 0-255
z = z * 255 / np.max(z)
z = np.uint8(z)
z = z[0, :, :]

# img = Image.fromarray(z) #convert to image
import matplotlib.pyplot as plt

plt.imshow(z)
plt.savefig(f'{NAME}.png', dpi=200)
