import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data

# argparser
parser = argparse.ArgumentParser(
    parser.add_argument("--dataroot", type=str, default="./datasets/human2anime")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--out", default="./output"))
args = parser.parse_args()

# Dataset and dataloader
data_transform = transforms.Compose([
                 transforms.Resize(int(128 * 1.12), Image.BICUBIC),
                 transforms.RandomCrop(128),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor()])

trainset = datasets.ImageFolder(root='human2anime', transform=data_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

# set device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
