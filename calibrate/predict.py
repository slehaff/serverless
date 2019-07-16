from keras import layers
import numpy as np
import os
import cv2
# import boto3models/cnn0a-173-model50.h5
import pandas as pd
# from sagemaker import get_execution_role

import keras
from keras.models import Sequential, Model, Input
from keras.layers import Conv2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from keras.engine.topology import Layer
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Add

IMWIDTH = 500
IMHEIGHT = 300


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = np.asarray(img)
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img - np.min(img)
    img = img/np.max(img)
    return img


def in_to_array(folder_path, imagename, array):
    myfile = folder_path + imagename
    img = cv2.imread(myfile).astype(np.float32)
    inp_img = make_grayscale(img)
    img = normalize_image255(inp_img)
    for j in range(0, IMWIDTH, 1):
        for i in range(0, IMHEIGHT, 1):
            sample = [i, j, img[i, j]]
            array.append(sample)
    return


def predict_image(inpfile, outfile):
    img = cv2.imread(inpfile).astype(np.float32)
    inp_img = make_grayscale(img)
    img = normalize_image255(inp_img)
    for j in range(0, IMWIDTH, 1):
        print(j)
        for i in range(0, IMHEIGHT, 1):
            sample = [i, j, img[i, j]]
            img[i, j] = DB_predict(sample)*125
    cv2.imwrite(outfile, img)


def out_to_array(z, array):
    count = IMWIDTH*IMHEIGHT
    sample = 0
    for i in range(0, count, 1):
        sample = z
        array.append(sample)
    return


def load_model():
    model = keras.models.load_model(
        'models/zcnn1.h5')
    model.summary()
    return(model)


model = load_model()


def DB_predict(x):
    predicted_sample = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_sample = predicted_sample.squeeze()

    return(predicted_sample)


infile = '/home/samir/serverless/inphi/'+'unwrap.png'
outfile = '/home/samir/serverless/inphi/'+'zunwrap.png'
predict_image(infile, outfile)

# input_samples = []
# output_samples = []

# in_to_array('/home/samir/serverless/inphi/', 'unwrap.png', input_samples)
# print(len(input_samples))
# for i in range(len(input_samples)):
#     if (i/10000 % 1) == 0:
#         print('i:', i)
#     output_samples.append(DB_predict(input_samples[i]))

# print(len(output_samples))
# print(output_samples[1500:1600])

# myfile= '/home/samir/serverless/inphi/'+'unwrap.png'
# img = cv2.imread(myfile).astype(np.float32)
# inp_img = make_grayscale(img)
# img = normalize_image255(inp_img)
