#! /usr/bin/env python3

import torch
from utils import ResidualBlock, cyclegan_loss


class CycleGAN(torch.nn.Module):

    # original network is:
    # c7s1-64,d128,d256,R256,R256,R256,
    # R256,R256,R256,u128,u64,c7s1-3

    def __init__(self, num_residual_blocks=6, size=256):
        super().__init__()

        self.activation = torch.nn.ReLU()

        # c7s1-64
        self.conv1 = torch.nn.Conv2d(3,
                                     64,
                                     kernel_size=7,
                                     stride=1,
                                     padding=3,
                                     padding_mode='reflect',
                                     )
        self.norm_64 = torch.nn.InstanceNorm2d(num_features=64)

        # d128
        self.conv2 = torch.nn.Conv2d(64,
                                     128,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     padding_mode='reflect',
                                     )
        self.norm_128 = torch.nn.InstanceNorm2d(num_features=128)

        # d256
        self.conv3 = torch.nn.Conv2d(128,
                                     256,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     padding_mode='reflect',
                                     )
        self.norm_256 = torch.nn.InstanceNorm2d(num_features=256)

        # R's
        self.residual_blocks = torch.nn.ModuleList()

        for _ in range(num_residual_blocks):
            residual_block = ResidualBlock(size)

            self.residual_blocks.append(residual_block)

        # u128
        # somehow, i think this is correct
        self.frac_conv1 = torch.nn.ConvTranspose2d(256,
                                                   128,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1,
                                                   )

        # u64
        self.frac_conv2 = torch.nn.ConvTranspose2d(128,
                                                   64,
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1,
                                                   output_padding=1,
                                                   )

        # c7s1-3
        self.conv4 = torch.nn.Conv2d(64,
                                     3,
                                     kernel_size=7,
                                     stride=1,
                                     padding=3,
                                     padding_mode='reflect',
                                     )
        self.norm_3 = torch.nn.InstanceNorm2d(num_features=3)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm_64(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm_128(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.norm_256(x)
        x = self.activation(x)

        for res_block in self.residual_blocks:
            x = res_block(x)

        x = self.frac_conv1(x)
        x = self.norm_128(x)
        x = self.activation(x)

        x = self.frac_conv2(x)
        x = self.norm_64(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.norm_3(x)
        x = self.activation(x)

        return x


class PatchGAN(torch.nn.Module):

    # based off the paper description
    # original network is:
    # C64-C128-C256-C512

    def __init__(self):
        super().__init__()

        self.activation = torch.nn.LeakyReLU(negative_slope=0.2)
        
        # C64
        self.conv1 = torch.nn.Conv2d(3,
                                     64,
                                     kernel_size=4,
                                     stride=2,
                                     )
        # skip the norm for this layer

        # C128
        self.conv2 = torch.nn.Conv2d(64,
                                     128,
                                     kernel_size=4,
                                     stride=2,
                                     )
        self.norm_128 = torch.nn.InstanceNorm2d(num_features=128)

        # C256
        self.conv3 = torch.nn.Conv2d(128,
                                     256,
                                     kernel_size=4,
                                     stride=2,
                                     )
        self.norm_256 = torch.nn.InstanceNorm2d(num_features=256)

        # C512
        self.conv4 = torch.nn.Conv2d(256,
                                     512,
                                     kernel_size=4,
                                     stride=2,
                                     )
        self.norm_512 = torch.nn.InstanceNorm2d(num_features=512)

        # produce 1-dimensional output

        self.conv_final = torch.nn.Conv2d(512,
                                          1,
                                          kernel_size=2,
                                          )

        # sigmoid at the end, according to https://arxiv.org/pdf/1611.07004.pdf

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm_128(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.norm_256(x)
        x = self.activation(x)

        x = self.conv4(x)
        x = self.norm_512(x)
        x = self.activation(x)

        x = self.conv_final(x)

        x = self.sigmoid(x)

        return x.squeeze()


if __name__ == '__main__':
    model = CycleGAN()

    t = torch.rand(size=(1, 3, 128, 128))

    output = model(t)

    print(output.shape)

    model = PatchGAN()

    t = torch.rand(size=(5, 3, 70, 70))

    output = model(t)

    print(output.shape)
