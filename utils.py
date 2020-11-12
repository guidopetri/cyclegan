#! /usr/bin/env python3

"""
Code for CycleGAN loss functions and the basic residual blocks that make up
the network.
"""

import torch


class AdversarialLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()


    def forward(self, discriminator, output, target):

        first = torch.log(discriminator(target))
        second = torch.log(1 - discriminator(output))

        return first + second


class CycleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, output, target):
        return self.l1_loss(output, target)


class IdentityLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss()

    def forward(self, output, target):
        return self.l1_loss(output, target)
