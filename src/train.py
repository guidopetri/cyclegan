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


## TODO: add arguments for the paths to weights for each model if continuing or using pre-trained
# argparser
parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="../datasets", help="path to folder containing datasets")
parser.add_argument("--dataset", type=str, default="human2anime", help="name of dataset (default: 'human2anime')")
parser.add_argument("--n-epochs", type=int, default=200, help="total number of training epochs")
parser.add_argument('--decay-epoch', type=int, default=100, help="epoch to start linearly decaying learning rate")
parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--image-size", type=int, default=128, help="image size")
parser.add_argument("--out", type=str, default="./output", help="output path")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--log-step", type=int, default=100, help="frequency to log progress")
parser.add_argument("--print-freq", type=int, default=100, help="frequency to print images")
parser.add_argument("--manualSeed", type=int, help="seed for training")
args = parser.parse_args()

unique_dir = args.n_epochs + args.batch_size + args.lr + args.image_size

# create directories for outputs
try:
    os.makedirs(os.path.join(args.out, args.dataset, unique_dir, "A"))
    os.makedirs(os.path.join(args.out, args.dataset, unique_dir, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join(args.out, args.dataset, "weights"))
except OSError:
    pass

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
                    transforms.Resize(int(128 * 1.12), Image.BICUBIC),
                    transforms.RandomCrop(128),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])

# create dataset
dataset = ImageDataset(root=os.path.join(args.dataroot, args.dataset),
                       transform=data_transform,
                       unaligned=True)

# create dataloader (note: pin_memory=True makes transferring samples to GPU faster)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('Network initialized with weights sampled from N(0,0.02).')
    net.apply(init_func)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
        
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.n_epochs - self.decay_epoch)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create models
g_AB = model.CycleGAN().to(device)
g_BA = model.CycleGAN().to(device)
d_A = model.PatchGAN().to(device)
d_B = model.PatchGAN().to(device)

# initialize weights      
g_AB.apply(weights_init)
g_BA.apply(weights_init)
d_A.apply(weights_init)
d_B.apply(weights_init)

## TODO: add code to load state dicts if continuing training with weights

# optimizers
optimizer_g = torch.optim.Adam(itertools.chain(g_AB.parameters(), g_BA.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(itertools.chain(d_A.parameters(), d_B.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))

# learning rate schedulers
g_lr_scheduler = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)
d_lr_scheduler = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=LambdaLR(args.n_epochs, 0, args.decay_epoch).step)

# loss functions
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# image buffers
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# training loop
for epoch in range(0, args.n_epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
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

        # Update progress bar
        progress_bar.set_description(
            f"[{epoch}/{args.n_epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(dis_A_loss + dis_B_loss).item():.4f} "
            f"Loss_G: {gen_loss.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"loss_G_GAN: {(gan_loss_AB + gan_loss_BA).item():.4f} "
            f"loss_G_cycle: {(cycle_loss_ABA + cycle_loss_BAB).item():.4f}")
        
        # save output images
        if i % args.print_freq == 0:
            vutils.save_image(real_img_A, f"{args.out}/{args.dataset}/{unique_dir}/A/real_samples_{epoch}.png",
                              normalize=True)
            vutils.save_image(real_img_B,
                              f"{args.out}/{args.dataset}/{unique_dir}/B/real_samples_{epoch}.png",
                              normalize=True)

            fake_img_A = 0.5 * (g_BA(real_img_B).data + 1.0)
            fake_img_B = 0.5 * (g_AB(real_img_A).data + 1.0)

            vutils.save_image(fake_img_A.detach(),
                              f"{args.out}/{args.dataset}/{unique_dir}/A/fake_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(fake_img_B.detach(),
                              f"{args.out}/{args.dataset}/{unique_dir}/B/fake_samples_epoch_{epoch}.png",
                              normalize=True)
    
    # save weights
    torch.save(g_AB.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/g_AB_epoch_{epoch}.pth")
    torch.save(g_BA.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/g_BA_epoch_{epoch}.pth")
    torch.save(d_A.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/d_A_epoch_{epoch}.pth")
    torch.save(d_B.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/d_B_epoch_{epoch}.pth")

    # Update learning rates
    g_lr_scheduler.step()
    d_lr_scheduler.step()

# save final weights
torch.save(g_AB.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/g_AB_epoch_{epoch}.pth")
torch.save(g_BA.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/g_BA_epoch_{epoch}.pth")
torch.save(d_A.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/d_A_epoch_{epoch}.pth")
torch.save(d_B.state_dict(), f"{args.out}/{args.dataset}/{unique_dir}/weights/d_B_epoch_{epoch}.pth")