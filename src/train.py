import model
import utils
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import init
from pathlib import Path
import torch.nn.functional as F
import itertools
import random
import glob
import os
from tqdm import tqdm
import torchvision.utils as vutils


# argparser
parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="../datasets", help="path to folder containing datasets")
parser.add_argument("--dataset", type=str, default="human2anime", help="name of dataset (default: 'human2anime')")
parser.add_argument("--n-epochs", type=int, default=200, help="total number of training epochs")
parser.add_argument("--decay-epoch", type=int, default=100, help="epoch to start linearly decaying learning rate")
parser.add_argument("--starting-epoch", type=int, default=0, help='epoch of the weights to load')
parser.add_argument("--image-size", type=int, default=256, help='pixel-size of the image')
parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--residual-blocks", type=int, default=9, help="number of residual blocks to use in generator")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--out", type=str, default="./output", help="output path")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--num-workers", type=int, default=4, help='number of workers for dataloader')
parser.add_argument("--input_nc", type=int, default=3, help='number of channels of input data')
parser.add_argument("--output_nc", type=int, default=3, help='number of channels of output data')
parser.add_argument("--log-freq", type=int, default=200, help='frequency of printing losses to stdout')
parser.add_argument("--visdom-freq", type=int, default=1000, help='frequency of showing training results in visdom')
parser.add_argument("--save-freq", type=int, default=1000, help="frequency to save images")
parser.add_argument("--visdom", type=utils.str2bool, nargs='?', const=True, default=False,
                    help='true/false value indicating whether to use visdom')
parser.add_argument("--save-images", type=utils.str2bool, nargs='?', const=True, default=True,
                    help='true/false value indicating whether to save images generated during training')
parser.add_argument("--verbose", type=utils.str2bool, nargs='?', const=True, default=True,
                    help='true/false value indicating whether to use tqdm')
parser.add_argument("--log", type=utils.str2bool, nargs='?', const=True, default=False,
                    help='true/false value indicating whether to log losses during training if not using tqdm')
parser.add_argument("--save-last-only", type=utils.str2bool, nargs='?', const=True, default=True,
                    help='true/false value indicating whether to only save latest weights')
parser.add_argument("--name", type=str, help='the unique directory name for each experiment')
parser.add_argument("--checkpoints_dir", type=str, default='./checkpoints', help='models are saved here')
parser.add_argument("--manualSeed", type=int, help="seed for training")
args = parser.parse_args()

# unique dir for saving outputs and weights
if args.name is None:
    unique_dir = f'{args.n_epochs}{args.batch_size}{args.lr}{args.image_size}'
else:
    unique_dir = args.name

print(f'Experiment name: {unique_dir}')
print(f'Start visdom server: {args.visdom}')

# create directories for outputs
try:
    os.makedirs(os.path.join(args.out, args.dataset, unique_dir, "A"))
    os.makedirs(os.path.join(args.out, args.dataset, unique_dir, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", args.dataset, unique_dir))
except OSError:
    pass

# directory for weights
weights_dir = os.path.join("./weights", args.dataset, unique_dir)

# set random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

# custom dataset class
class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, f"{mode}/A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, f"{mode}/B") + "/*.*"))

    def __getitem__(self, index):
        img_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            img_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            img_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

# define transformations
data_transform = transforms.Compose([
                    transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                    transforms.RandomCrop(args.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# create dataset
dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                       transform=data_transform,
                       unaligned=True)

# create dataloader (note: pin_memory=True makes transferring samples to GPU faster)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
print(f'Length of Dataloader: {len(dataloader)}')

# set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# enable cuDNN benchmark for performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# create models
g_AB = model.CycleGAN(residual_blocks=args.residual_blocks).to(device)
g_BA = model.CycleGAN(residual_blocks=args.residual_blocks).to(device)
d_A = model.PatchGAN().to(device)
d_B = model.PatchGAN().to(device)

# initialize weights
g_AB.apply(utils.weights_init)
g_BA.apply(utils.weights_init)
d_A.apply(utils.weights_init)
d_B.apply(utils.weights_init)

# optimizers
optimizer_g = torch.optim.Adam(itertools.chain(g_AB.parameters(), g_BA.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(itertools.chain(d_A.parameters(), d_B.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))

# learning rate schedulers
g_lr_scheduler = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=utils.LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step)
d_lr_scheduler = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=utils.LambdaLR(args.n_epochs, args.starting_epoch, args.decay_epoch).step)

# loss functions
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# image buffers
fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()

# Logger Image Plots
if args.visdom:
    logger = utils.Logger(args.n_epochs, len(dataloader))

# load state dicts
if args.starting_epoch > 0:
    g_AB.load_state_dict(torch.load(os.path.join(weights_dir, f'g_AB_epoch_{args.starting_epoch}.pth')))
    g_BA.load_state_dict(torch.load(os.path.join(weights_dir, f'g_BA_epoch_{args.starting_epoch}.pth')))
    d_A.load_state_dict(torch.load(os.path.join(weights_dir, f'd_A_epoch_{args.starting_epoch}.pth')))
    d_B.load_state_dict(torch.load(os.path.join(weights_dir, f'd_B_epoch_{args.starting_epoch}.pth')))
    optimizer_g.load_state_dict(torch.load(os.path.join(weights_dir, f'optimizer_g_{args.starting_epoch}.pth')))
    optimizer_d.load_state_dict(torch.load(os.path.join(weights_dir, f'optimizer_d_{args.starting_epoch}.pth')))

# training loop
for epoch in range(args.starting_epoch, args.n_epochs):
    # if args.verbose is True, enable tqdm progress bar
    if args.verbose:
        enumerator = tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        enumerator = enumerate(dataloader)

    for i, data in enumerator:
        # get images
        real_img_A = data["A"].to(device)
        real_img_B = data["B"].to(device)
        batch_size = real_img_A.size(0)

        # real data label is 1, fake data label is 0
        real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)


        '''Generator Computations'''

        optimizer_g.zero_grad()

        ## Identity losses
        # g_BA(A) should equal A if real A is passed
        identity_img_A = g_BA(real_img_A)
        loss_identity_A = identity_loss(identity_img_A, real_img_A) * 5.0
        # g_AB(B) should equal B if real B is passed
        identity_img_B = g_AB(real_img_B)
        loss_identity_B = identity_loss(identity_img_B, real_img_B) * 5.0

        ## GAN losses
        # GAN loss d_A(g_A(A))
        fake_img_A = g_BA(real_img_B)
        fake_output_A = d_A(fake_img_A)
        gan_loss_BA = adversarial_loss(fake_output_A, real_label)
        # GAN loss d_B(d_B(B))
        fake_img_B = g_AB(real_img_A)
        fake_output_B = d_B(fake_img_B)
        gan_loss_AB = adversarial_loss(fake_output_B, real_label)

        ## Cycle losses
        # reconstructed A vs real A; A vs g_BA(g_AB(A))
        recovered_img_A = g_BA(fake_img_B)
        cycle_loss_ABA = cycle_loss(recovered_img_A, real_img_A) * 10.0
        # reconstructed B vs real B; B vs g_AB(g_BA(B))
        recovered_img_B = g_AB(fake_img_A)
        cycle_loss_BAB = cycle_loss(recovered_img_B, real_img_B) * 10.0

        # Combined generator losses
        gen_loss = loss_identity_A + loss_identity_B + gan_loss_AB + gan_loss_BA + cycle_loss_ABA + cycle_loss_BAB

        # Calculate generator gradients
        gen_loss.backward()

        # Update generator weights
        optimizer_g.step()


        '''Discriminator Computations'''

        # Set discriminator gradients to zero
        optimizer_d.zero_grad()

        ## Discriminator A losses

        # Real A image loss
        real_output_A = d_A(real_img_A)
        d_A_real_loss = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_img_A = fake_A_buffer.push_and_pop(fake_img_A)
        fake_output_A = d_A(fake_img_A.detach())
        d_A_fake_loss = adversarial_loss(fake_output_A, fake_label)

        # Combined discriminator A loss
        dis_A_loss = (d_A_real_loss + d_A_fake_loss)/2

        # Calculate discriminator A gradients
        dis_A_loss.backward()

        ## Discriminator B losses

        # Real B image loss
        real_output_B = d_B(real_img_B)
        d_B_real_loss = adversarial_loss(real_output_B, real_label)

        # Fake A image loss
        fake_img_B = fake_B_buffer.push_and_pop(fake_img_B)
        fake_output_B = d_B(fake_img_B.detach())
        d_B_fake_loss = adversarial_loss(fake_output_B, fake_label)

        # Combined discriminator A loss
        dis_B_loss = (d_B_real_loss + d_B_fake_loss)/2

        # Calculate discriminator A gradients
        dis_B_loss.backward()

        ## Update discriminator weights
        optimizer_d.step()

        # Update tqdm progress bar, or print to screen
        if args.verbose:
            enumerator.set_description(
                f"[{epoch}/{args.n_epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_D: {(dis_A_loss + dis_B_loss).item():.4f} "
                f"Loss_G: {gen_loss.item():.4f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                f"loss_G_GAN: {(gan_loss_AB + gan_loss_BA).item():.4f} "
                f"loss_G_cycle: {(cycle_loss_ABA + cycle_loss_BAB).item():.4f}")
        elif args.log:
            if i % args.log_freq == 0:
                print(f"[{epoch}/{args.n_epochs - 1}][{i}/{len(dataloader) - 1}]"
                      f"Loss_D: {(dis_A_loss + dis_B_loss).item():.4f} "
                      f"Loss_G: {gen_loss.item():.4f} "
                      f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                      f"loss_G_GAN: {(gan_loss_AB + gan_loss_BA).item():.4f} "
                      f"loss_G_cycle: {(cycle_loss_ABA + cycle_loss_BAB).item():.4f}")

        # save output images
        if args.save_images:
            if i % args.save_freq == 0:
                vutils.save_image(real_img_A, f"{args.out}/{args.dataset}/{unique_dir}/A/real_samples_{epoch}.png",
                                  normalize=True)
                vutils.save_image(real_img_B, f"{args.out}/{args.dataset}/{unique_dir}/B/real_samples_{epoch}.png",
                                  normalize=True)

                fake_img_A = 0.5 * (g_BA(real_img_B).data + 1.0)
                fake_img_B = 0.5 * (g_AB(real_img_A).data + 1.0)

                vutils.save_image(fake_img_A.detach(), f"{args.out}/{args.dataset}/{unique_dir}/A/fake_samples_epoch_{epoch}.png",
                                  normalize=True)
                vutils.save_image(fake_img_B.detach(), f"{args.out}/{args.dataset}/{unique_dir}/B/fake_samples_epoch_{epoch}.png",
                                  normalize=True)

        # Visdom updates
        loss_dict = {'G_loss': gen_loss, 'G_Idt_loss': (loss_identity_A + loss_identity_B), 'G_GAN_loss': (gan_loss_AB + gan_loss_BA),
                        'G_cycle_loss': (cycle_loss_ABA + cycle_loss_BAB), 'D_loss': (dis_A_loss + dis_B_loss)}
        if args.visdom:
            if i % args.visdom_freq == 0:
                logger.log(loss_dict, images={'real_A': real_img_A, 'real_B': real_img_B, 'fake_A': fake_img_A, 'fake_B': fake_img_B})

    # save weights
#     if args.save_last_only:
#         torch.save(g_AB.state_dict(), f"weights/{args.dataset}/{unique_dir}/g_AB_epoch_latest.pth")
#         torch.save(g_BA.state_dict(), f"weights/{args.dataset}/{unique_dir}/g_BA_epoch_latest.pth")
#         torch.save(d_A.state_dict(), f"weights/{args.dataset}/{unique_dir}/d_A_epoch_latest.pth")
#         torch.save(d_B.state_dict(), f"weights/{args.dataset}/{unique_dir}/d_B_epoch_latest.pth")
    torch.save(g_AB.state_dict(), f"weights/{args.dataset}/{unique_dir}/g_AB_epoch_{epoch}.pth")
    torch.save(g_BA.state_dict(), f"weights/{args.dataset}/{unique_dir}/g_BA_epoch_{epoch}.pth")
    torch.save(d_A.state_dict(), f"weights/{args.dataset}/{unique_dir}/d_A_epoch_{epoch}.pth")
    torch.save(d_B.state_dict(), f"weights/{args.dataset}/{unique_dir}/d_B_epoch_{epoch}.pth")
    torch.save(optimizer_g.state_dict(), f"weights/{args.dataset}/{unique_dir}/optimizer_g_{epoch}.pth")
    torch.save(optimizer_d.state_dict(), f"weights/{args.dataset}/{unique_dir}/optimizer_d_{epoch}.pth")

    if args.save_last_only:
        if (epoch - args.starting_epoch ) >= 5:
            os.remove(f"weights/{args.dataset}/{unique_dir}/g_AB_epoch_{epoch-5}.pth")
            os.remove(f"weights/{args.dataset}/{unique_dir}/g_BA_epoch_{epoch-5}.pth")
            os.remove(f"weights/{args.dataset}/{unique_dir}/d_A_epoch_{epoch-5}.pth")
            os.remove(f"weights/{args.dataset}/{unique_dir}/d_B_epoch_{epoch-5}.pth")
            os.remove(f"weights/{args.dataset}/{unique_dir}/optimizer_g_{epoch-5}.pth")
            os.remove(f"weights/{args.dataset}/{unique_dir}/optimizer_d_{epoch-5}.pth")

    # Update learning rates
    g_lr_scheduler.step()
    d_lr_scheduler.step()

# save final weights
torch.save(g_AB.state_dict(), f"weights/{args.dataset}/{unique_dir}/g_AB_epoch_{epoch}.pth")
torch.save(g_BA.state_dict(), f"weights/{args.dataset}/{unique_dir}/g_BA_epoch_{epoch}.pth")
torch.save(d_A.state_dict(), f"weights/{args.dataset}/{unique_dir}/d_A_epoch_{epoch}.pth")
torch.save(d_B.state_dict(), f"weights/{args.dataset}/{unique_dir}/d_B_epoch_{epoch}.pth")
torch.save(optimizer_g.state_dict(), f"weights/{args.dataset}/{unique_dir}/optimizer_g_{epoch}.pth")
torch.save(optimizer_d.state_dict(), f"weights/{args.dataset}/{unique_dir}/optimizer_d_{epoch}.pth")
