# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2

path = './scannn/data/train_im_folder/'
files = os.listdir(path + 'inputstrip/')
for f in files:
    os.rename(path + 'inputstrip/' + f, path + 'inputstrip/' + f[5:])
