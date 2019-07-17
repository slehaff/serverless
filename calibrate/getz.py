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
import os

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


def out_to_array(z, array):
    count = IMWIDTH*IMHEIGHT
    sample = 0
    for i in range(0, count, 1):
        sample = z
        array.append(sample)
    return


def compile_model(model):
    # model = Model(input_image, output_image)
    sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mse'])
    model.summary()
    return(model)


# Load and pre-process the training data
input_samples = []
output_samples = []

z1 = 0  # 0/44 0mm from reference
z2 = 0.159  # 7/44, 7 mm from reference
z3 = 0.579  # 25.5/44mm 25.5mm from reference
z4 = 0.863  # 38/44, 38mm from reference
z5 = 1.0  # 44mm from reference

# in_to_array('/home/samir/serverless/inphi/', 'unwrap1.png', input_samples)
in_to_array('/home/samir/serverless/inphi/', 'unwrap2.png', input_samples)
in_to_array('/home/samir/serverless/inphi/', 'unwrap3.png', input_samples)
in_to_array('/home/samir/serverless/inphi/', 'unwrap4.png', input_samples)
in_to_array('/home/samir/serverless/inphi/', 'unwrap5.png', input_samples)
# out_to_array(z1, output_samples)
out_to_array(z2, output_samples)
out_to_array(z3, output_samples)
out_to_array(z4, output_samples)
out_to_array(z5, output_samples)

print(input_samples[770:795])
print(output_samples[160000:160020])
# print('Length:', len(input_samples))
# print('Length:', len(output_samples))
# print(input_samples.size)

# Expand the image dimension to conform with the shape required by keras and tensorflow, inputshape=(..., h, w, nchannels).
input_samples = np.expand_dims(input_samples, -1)
output_samples = np.expand_dims(output_samples, -1)
input = Input(shape=(3, 1))
print('input_shape:', input_samples.shape)
print('output_shape:', output_samples.shape)


flatten_layer = Flatten()  # instantiate the layer
x0 = flatten_layer(input)       # call it on the given tensor
x1 = Dense(12, kernel_initializer='normal', activation='relu')(x0)
x2 = Dense(24, activation='relu')(x1)
# x3 = Dropout(.5)(x1)
# x4 = Dense(24, activation='relu')(x1)
x5 = Dense(8, activation='relu')(x2)
print('x5shape:', x5.shape)

zOutput = Dense(1, activation='linear')(x5)

z_model = Model(input, zOutput)
# z_model.summary()


number_of_epochs = 10
loss = []
val_loss = []


def fct_train():
    for epoch in range(number_of_epochs):
        print('epoch #:', epoch)
        history_temp = model.fit(input_samples, output_samples,
                                 batch_size=100,
                                 epochs=1,
                                 validation_split=0.2,
                                 )
        loss.append(history_temp.history['loss'][0])
        val_loss.append(history_temp.history['val_loss'][0])
        # convweights.append(model.layers[0].get_weights()[0].squeeze())


compile_model(z_model)
model = z_model


def load_model():
    model = keras.models.load_model(
        'models/zcnn1.h5')
    model.summary()
    return(model)


# model = load_model()

fct_train()


def plot():
    # Plot the training and validation losses
    plt.close('all')

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training loss', 'Validation loss'], loc='upper right')
    plt.show()
    plt.savefig('trainingvalidationlossgx.png')


plot()


def DB_predict(x):
    predicted_sample = model.predict(np.array([np.expand_dims(x, -1)]))
    predicted_sample = predicted_sample.squeeze()

    return(predicted_sample)


model.save('models/zcnn1.h5')

x = [171, 2, 0.039215688]
print('predicted:', DB_predict(x))
x = [189, 2, 0.039215688]
print('predicted:', DB_predict(x))
