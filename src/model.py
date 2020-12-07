#! /usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F
from utils import cyclegan_loss


class ResidualBlock(nn.Module):

    def __init__(self, size):
        super(ResidualBlock, self).__init__()

        # two 3x3 conv layers (with same number of filters: 256)
        # batch norm? or instance norm? for batch size=1, it's the same
        # but which to use... probably instance norm is best for
        # reimplementation

        # uses ReLU internally, and does:
        # input - conv - norm - relu - conv - norm - addition - relu - output

        # reference:
        # http://torch.ch/blog/2016/02/04/resnets.html

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(size, size, 3),
                                   nn.InstanceNorm2d(size),
                                   nn.ReLU(inplace=True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(size, size, 3),
                                   nn.InstanceNorm2d(size))

    def forward(self, x):
        return x + self.block(x)

    
class CycleGAN(nn.Module):

    # original network is:
    # c7s1-64,d128,d256,R256,R256,R256,
    # R256,R256,R256,u128,u64,c7s1-3

    def __init__(self):
        super(CycleGAN, self).__init__()

        self.main = nn.Sequential(           
            # initial conv block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # downsampling layers
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            # residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            
            # upsampling layers
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            # output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):     
        return self.main(x)


class PatchGAN(nn.Module):

    # based off the paper description
    # original network is:
    # C64-C128-C256-C512

    def __init__(self):
        super(PatchGAN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    
# import torch
# from utils import ResidualBlock, cyclegan_loss


# class CycleGAN(torch.nn.Module):

#     # original network is:
#     # c7s1-64,d128,d256,R256,R256,R256,
#     # R256,R256,R256,u128,u64,c7s1-3

#     def __init__(self, num_residual_blocks=6, size=256):
#         super().__init__()

#         self.activation = torch.nn.ReLU()

#         # c7s1-64
#         self.conv1 = torch.nn.Conv2d(3,
#                                      64,
#                                      kernel_size=7,
#                                      stride=1,
#                                      padding=3,
#                                      padding_mode='reflect',
#                                      )
#         self.norm_64 = torch.nn.InstanceNorm2d(num_features=64)

#         # d128
#         self.conv2 = torch.nn.Conv2d(64,
#                                      128,
#                                      kernel_size=3,
#                                      stride=2,
#                                      padding=1,
#                                      padding_mode='reflect',
#                                      )
#         self.norm_128 = torch.nn.InstanceNorm2d(num_features=128)

#         # d256
#         self.conv3 = torch.nn.Conv2d(128,
#                                      256,
#                                      kernel_size=3,
#                                      stride=2,
#                                      padding=1,
#                                      padding_mode='reflect',
#                                      )
#         self.norm_256 = torch.nn.InstanceNorm2d(num_features=256)

#         # R's
#         self.residual_blocks = torch.nn.ModuleList()

#         for _ in range(num_residual_blocks):
#             residual_block = ResidualBlock(size)

#             self.residual_blocks.append(residual_block)

#         # u128
#         # somehow, i think this is correct
#         self.frac_conv1 = torch.nn.ConvTranspose2d(256,
#                                                    128,
#                                                    kernel_size=3,
#                                                    stride=2,
#                                                    padding=1,
#                                                    output_padding=1,
#                                                    )

#         # u64
#         self.frac_conv2 = torch.nn.ConvTranspose2d(128,
#                                                    64,
#                                                    kernel_size=3,
#                                                    stride=2,
#                                                    padding=1,
#                                                    output_padding=1,
#                                                    )

#         # c7s1-3
#         self.conv4 = torch.nn.Conv2d(64,
#                                      3,
#                                      kernel_size=7,
#                                      stride=1,
#                                      padding=3,
#                                      padding_mode='reflect',
#                                      )
#         self.norm_3 = torch.nn.InstanceNorm2d(num_features=3)

#     def forward(self, x):
        
#         x = self.conv1(x)
#         x = self.norm_64(x)
#         x = self.activation(x)

#         x = self.conv2(x)
#         x = self.norm_128(x)
#         x = self.activation(x)

#         x = self.conv3(x)
#         x = self.norm_256(x)
#         x = self.activation(x)

#         for res_block in self.residual_blocks:
#             x = res_block(x)

#         x = self.frac_conv1(x)
#         x = self.norm_128(x)
#         x = self.activation(x)

#         x = self.frac_conv2(x)
#         x = self.norm_64(x)
#         x = self.activation(x)

#         x = self.conv4(x)
#         x = self.norm_3(x)
#         x = self.activation(x)

#         return x


# class PatchGAN(torch.nn.Module):

#     # based off the paper description
#     # original network is:
#     # C64-C128-C256-C512

#     def __init__(self):
#         super().__init__()

#         self.activation = torch.nn.LeakyReLU(negative_slope=0.2)
        
#         # C64
#         self.conv1 = torch.nn.Conv2d(3,
#                                      64,
#                                      kernel_size=4,
#                                      stride=2,
#                                      )
#         # skip the norm for this layer

#         # C128
#         self.conv2 = torch.nn.Conv2d(64,
#                                      128,
#                                      kernel_size=4,
#                                      stride=2,
#                                      )
#         self.norm_128 = torch.nn.InstanceNorm2d(num_features=128)

#         # C256
#         self.conv3 = torch.nn.Conv2d(128,
#                                      256,
#                                      kernel_size=4,
#                                      stride=2,
#                                      )
#         self.norm_256 = torch.nn.InstanceNorm2d(num_features=256)

#         # C512
#         self.conv4 = torch.nn.Conv2d(256,
#                                      512,
#                                      kernel_size=4,
#                                      stride=2,
#                                      )
#         self.norm_512 = torch.nn.InstanceNorm2d(num_features=512)

#         # produce 1-dimensional output
#         # this is a 5x5 output when input is 128x128 in the default
#         # implementation, so we need to convert to a single value

#         self.conv_final = torch.nn.Conv2d(512,
#                                           1,
#                                           kernel_size=6,
#                                           )

#         # sigmoid at the end, according to https://arxiv.org/pdf/1611.07004.pdf

#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):

#         x = self.conv1(x)
#         x = self.activation(x)

#         x = self.conv2(x)
#         x = self.norm_128(x)
#         x = self.activation(x)

#         x = self.conv3(x)
#         x = self.norm_256(x)
#         x = self.activation(x)

#         x = self.conv4(x)
#         x = self.norm_512(x)
#         x = self.activation(x)

#         x = self.conv_final(x)

#         x = self.sigmoid(x)

#         return x.squeeze()


if __name__ == '__main__':
    model = CycleGAN()

    t = torch.rand(size=(1, 3, 128, 128))

    output = model(t)

    print(output.shape)

    model = PatchGAN()

    t = torch.rand(size=(5, 3, 128, 128))

    output = model(t)

    print(output.shape)
