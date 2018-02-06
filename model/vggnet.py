## https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def make_layers(self, in_shape, cfg, batch_norm):
        in_channels, height, width = in_shape

        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def __init__(self, in_shape, num_classes, cfg, batch_norm=False):
        super(VGG, self).__init__()
        in_channels, height, width = in_shape

        self.features = self.make_layers(cfg=cfg, in_shape=in_shape, batch_norm=batch_norm)
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x,output_size=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        logit = x
        prob = F.sigmoid(logit)
        return logit, prob

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    model = VGG(cfg=cfg['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    model = VGG(cfg=cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(pretrained=False, **kwargs):
    model = VGG(cfg=cfg['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    model = VGG(cfg=cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    model = VGG(cfg=cfg['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    model = VGG(cfg=cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    model = VGG(cfg=cfg['E'], **kwargs)
    return model

def vgg19_bn(**kwargs):
    model = VGG(cfg=cfg['E'], batch_norm=True, **kwargs)
    return model

