import model
import utils
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.utils.data
import random
import glob
import os
import torchvision.utils as vutils


# argparser
parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="../datasets", help="path to folder containing datasets")
parser.add_argument("--dataset", type=str, default="human2anime", help="name of dataset (default: 'human2anime')")
parser.add_argument("--image-size", type=int, default=256, help='pixel-size of the image')
parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--cuda", type=utils.str2bool, nargs='?', const=True, default=True,
                    help='true/false value indicating whether to use cuda')
parser.add_argument("--residual-blocks", type=int, default=9, help="number of residual blocks to use in generator")
parser.add_argument("--name", type=str, help='unique directory name for the experiment')
parser.add_argument("--resultdir", type=str, help='unique directory name for results, if different than experiment')
parser.add_argument("--weights-epoch", type=int, default=199, help='epoch of the weights to load')
parser.add_argument("--manualSeed", type=int, help="seed for training")
parser.add_argument("--acount", type=int, default=1040, help='number of images from domain A to process')
parser.add_argument("--bcount", type=int, default=1040, help='number of images from domain B to process')
args = parser.parse_args()

# unique dir for outputs, weights
if args.name is None:
    unique_dir = f'{args.n_epochs}{args.batch_size}{args.lr}{args.image_size}'
else:
    unique_dir = args.name

# unique dir for results
if args.resultdir is None:
    result_dir = unique_dir
else:
    result_dir = args.resultdir

print(f'Experiment name: {unique_dir}')

# create directories for results
try:
    os.makedirs(os.path.join("./results", args.dataset, result_dir, "A"))
    os.makedirs(os.path.join("./results", args.dataset, result_dir, "B"))
except OSError:
    pass

# directory for weights
weights_dir = os.path.join("./weights", args.dataset, unique_dir)

# define transformations
data_transform = transforms.Compose([
                    transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# create datasets
dataset = datasets.ImageFolder(root=os.path.join(args.dataroot, args.dataset, "test"),
                               transform=data_transform)

# create dataloader (note: pin_memory=True makes transferring samples to GPU faster)
dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=args.cuda)

# set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create models
g_AB = model.CycleGAN(residual_blocks=args.residual_blocks).to(device)
g_BA = model.CycleGAN(residual_blocks=args.residual_blocks).to(device)

# Load state dicts
g_AB.load_state_dict(torch.load(os.path.join(weights_dir, f'g_AB_epoch_{args.weights_epoch}.pth')))
g_BA.load_state_dict(torch.load(os.path.join(weights_dir, f'g_BA_epoch_{args.weights_epoch}.pth')))

# Set models to eval mode
g_AB.eval()
g_BA.eval()

# counts of images processed from each domain
A_counter = 0
B_counter = 0

print('Processing Images...')

# human to anime loop
for i, (data, target) in enumerate(dataloader):
    # break if max allowed images has been processed
    if (A_counter + B_counter) >= (args.acount + args.bcount):
        break

    # process human images
    if target==0:

        # condition
        if A_counter < args.acount:
            # get image
            real_img_A = data.to(device)

            # Generate output
            fake_img_B = 0.5 * (g_AB(real_img_A).data + 1.0)

            # Save image files
            vutils.save_image(real_img_A.detach(), f"./results/{args.dataset}/{result_dir}/A/image_{i}_real.png", normalize=True)
            vutils.save_image(fake_img_B.detach(), f"./results/{args.dataset}/{result_dir}/A/image_{i}_fake.png", normalize=True)

            # increment counter
            A_counter += 1

    # process anime images
    elif target==1:

        # condition
        if B_counter < args.bcount:
            # get image
            real_img_B = data.to(device)

            # Generate output
            fake_img_A = 0.5 * (g_BA(real_img_B).data + 1.0)

            # Save image files
            vutils.save_image(real_img_B.detach(), f"./results/{args.dataset}/{result_dir}/B/image_{i}_real.png", normalize=True)
            vutils.save_image(fake_img_A.detach(), f"./results/{args.dataset}/{result_dir}/B/image_{i}_fake.png", normalize=True)

            # increment counter
            B_counter += 1

    # print progress
    if ((i+1) % 100 == 0) and ((A_counter + B_counter) >= 100):
        print(f'{A_counter} human images processed, {B_counter} anime images processed')
print('Done')
