import torch.nn as nn
from model.bottleneck import Bottleneck


def net_layer_0():
    return [
        nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    ]


def net_layer_1():
    return [
        nn.BatchNorm2d(64, eps=1e-05, momentum=0, affine=True, track_running_stats=True)
    ]


def net_layer_2():
    return [nn.ReLU(inplace=True)]


def net_layer_3():
    return [
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    ]


def net_layer_4():
    return [
        Bottleneck(
            in_channels=64,
            out_channels=256,
            bottleneck_width=64,
            stride=1,
            downsample=nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
            ),
        ),
        Bottleneck(
            in_channels=256,
            out_channels=256,
            bottleneck_width=64,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=256,
            out_channels=256,
            bottleneck_width=64,
            stride=1,
            downsample=None,
        ),
    ]


def net_layer_5():
    return [
        Bottleneck(
            in_channels=256,
            out_channels=512,
            bottleneck_width=128,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512),
            ),
        ),
        Bottleneck(
            in_channels=512,
            out_channels=512,
            bottleneck_width=128,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=512,
            out_channels=512,
            bottleneck_width=128,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=512,
            out_channels=512,
            bottleneck_width=128,
            stride=1,
            downsample=None,
        ),
    ]


def net_layer_6():
    return [
        Bottleneck(
            in_channels=512,
            out_channels=1024,
            bottleneck_width=256,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(1024),
            ),
        ),
        Bottleneck(
            in_channels=1024,
            out_channels=1024,
            bottleneck_width=256,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=1024,
            out_channels=1024,
            bottleneck_width=256,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=1024,
            out_channels=1024,
            bottleneck_width=256,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=1024,
            out_channels=1024,
            bottleneck_width=256,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=1024,
            out_channels=1024,
            bottleneck_width=256,
            stride=1,
            downsample=None,
        ),
    ]


def net_layer_7():
    return [
        Bottleneck(
            in_channels=1024,
            out_channels=2048,
            bottleneck_width=512,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(2048),
            ),
        ),
        Bottleneck(
            in_channels=2048,
            out_channels=2048,
            bottleneck_width=512,
            stride=1,
            downsample=None,
        ),
        Bottleneck(
            in_channels=2048,
            out_channels=2048,
            bottleneck_width=512,
            stride=1,
            downsample=None,
        ),
    ]
