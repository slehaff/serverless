# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from scipy import misc
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


image = misc.imread('scannn/data/train_im_folder/input/image2.png')
print('type', type(image))
print(image.shape)
print(image.dtype)
image = rgb2gray(image)
plt.imshow(image, cmap='gray')
plt.show()
