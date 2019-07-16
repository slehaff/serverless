# -*- coding: utf-8 -*-
from torch.utils import data as D
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
import glob
import os
import numpy as np
import pandas as pd
import gc

import warnings
warnings.filterwarnings('ignore')

path = './scannn/data/train_im_folder/'


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ScanDS(D.Dataset):
    """
    A customized data loader.
    """

    def __init__(self, root):
        """ Intialize the dataset
        """
        self.input_filenames = []
        self.output_fielnames = []
        self.root = root
        self.transform = transforms.ToTensor()

        input_filenames = os.listdir(path + 'inputstrip/')
        for fn in input_filenames:
            self.input_filenames.append(fn)
        self.len = len(self.input_filenames)

        output_fielnames = os.listdir(path + 'outputstrip/')
        for fn in output_fielnames:
            self.output_fielnames.append(fn)
        self.len = len(self.output_fielnames)

    # You must override __getitem__ and __len__

    def __getitem__(self, index):
        """ Get input image and output image in tensor form
        """
        inpfname = path + 'inputstrip/' + str(index) + '.png'
        inpimage = Image.open(inpfname)
        print(inpfname)

        outfname = path + 'outputstrip/' + str(index + 1) + '.png'
        outimage = Image.open(outfname)
        print(outfname)

        return self.transform(inpimage)[0], self.transform(outimage)[0]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


scandata = ScanDS(path)
print(path)

# Use the torch dataloader to iterate through the dataset
loader = D.DataLoader(scandata, shuffle=False, num_workers=0)

# functions to show an image


def imshow(img):
    """
    Help Routine for image display!
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some images
dataiter = iter(loader)
images = dataiter.next()
for i in range(2, 10, 2):
    myimage = scandata.__getitem__(i)
    print(myimage[0].size(), myimage[1].size())
print('mylength:', scandata.len)


# show images
plt.figure(figsize=(16, 8))
imshow(torchvision.utils.make_grid(images))
plt.show()


class ScanNet(nn.Module):
    def __init__(self):
        super(ScanNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU())

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(50),
        #     nn.ReLU())
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(50, 1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU())

    def forward(self, x):
        """
        My forward function
        """
        output = self.layer1(x)

        # output = self.layer2(output)

        # output = self.layer3(output)

        return output


"""        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
            in_channels=50, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()"""


scannn = ScanNet().to(device)  # Innstantiate the nn model
train_set = scandata
train_loader = D.DataLoader(train_set, batch_size=10, shuffle=True)

sample = next(iter(train_set))
print('sample:', len(sample), type(sample))

inp, outp = train_set.__getitem__(4)
print(inp.shape, outp.shape)


def mse(t1, t2):
    diff = t1-t2
    print(diff.numel())
    return torch.sum(diff * diff) / diff.numel()


optimizer = torch.optim.SGD(scannn.parameters(), lr=0.01)
for epoch in range(500):
    y_pred = scannn.forward(inp)
    loss = mse(y_pred, outp)
    print('epoch: ', epoch, ' loss: ', loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
