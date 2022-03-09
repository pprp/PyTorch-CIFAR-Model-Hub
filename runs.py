# -*- coding: utf-8 -*-
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

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
        self.Norm = BasicRFB(256, 256, scale = 1.0, visual=2)
        
    def forward(self, x):
        print(x.shape)
        x = self.stem(x)
        s = self.Norm(x)
        return s


net = RFBFeature() #build the net

def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out') 
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0

net.Norm.apply(weights_init) #initial
input_shape = [32, 32, 3]

imgt = Image.open(r"./dd.jpg", mode="r")
imgt = imgt.resize((input_shape[0], input_shape[1]), Image.ANTIALIAS)
trans = transforms.Compose([transforms.ToTensor(),
			 transforms.Resize(32),
			 transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) 

x = trans(imgt)

x = x.unsqueeze(0)
    
# imgt = tile_pil_image(imgt, tile_factor=0, shade=True)

x = Variable(x, requires_grad=True) #input
out = net(x) #output

Zero_grad = torch.Tensor(1,256,10,10).zero_() #Zero_grad

Zero_grad[0][...][5][5] = 1 #set the middle pixel to 1.0

out.backward(Zero_grad) #backward

z = x.grad.data.cpu().numpy() #get input graident

print(z)
z = np.sum(np.abs(z), axis=1)

# z = np.mean(z,axis=1) #calculate mean by channels
# # z = np.array(z).mean(0).
# z /= z.max() 
# z += (np.abs(z) > 0) * 0.2 


#convert to 0-255
z = z * 255 / np.max(z)
z = np.uint8(z)
z = z[0,:,:]

# img = Image.fromarray(z) #convert to image
import matplotlib.pyplot as plt 

plt.imshow(z)
plt.savefig("out.png", dpi=200)