#! /usr/bin/env python3

"""
Code for CycleGAN loss functions and the basic residual blocks that make up
the network.
"""

import torch
import torch.nn.functional as F


def adversarial_loss(discriminator, output, target):
    first = torch.log(discriminator(target))
    second = torch.log(1 - discriminator(output))

    return first + second


def cycle_loss(G, F, x, y):
    first = F.l1_loss(F(G(x)), x)
    second = F.l1_loss(G(F(y)), y)
    return first + second


def identity_loss(G, F, x, y):
    first = F.l1_loss(G(y), y)
    second = F.l1_loss(F(x), x)
    return first + second


def cyclegan_loss(D_g, D_f, G, F, x, y):

    gan_loss_1 = adversarial_loss(D_g, G(x), y)
    gan_loss_2 = adversarial_loss(D_f, F(y), x)

    cycle = cycle_loss(G, F, x, y)
    identity = identity_loss(G, F, x, y)

    return gan_loss_1 + gan_loss_2 + cycle + identity


class ResidualBlock(torch.nn.Module):

    def __init__(self, size):
        super().__init__()

        # two 3x3 conv layers (with same number of filters: 256)
        # batch norm? or instance norm? for batch size=1, it's the same
        # but which to use... probably instance norm is best for
        # reimplementation

        # uses ReLU internally, and does:
        # input - conv - norm - relu - conv - norm - addition - relu - output

        # reference:
        # http://torch.ch/blog/2016/02/04/resnets.html

        self.size = size

        self.conv1 = torch.nn.Conv2d(self.size,
                                     self.size,
                                     kernel_size=3,
                                     )
        self.conv2 = torch.nn.Conv2d(self.size,
                                     self.size,
                                     kernel_size=3,
                                     )

        self.activation = torch.nn.ReLU()
        self.norm = torch.nn.InstanceNorm2d(num_features=self.size)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm(y)
        y = self.activation(y)

        y = self.conv2(y)
        y = self.norm(y)
        # the "residual" part
        y += x
        y = self.activation(y)

        return y
