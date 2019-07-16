import numpy as np
import os
import cv2

import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras import optimizers

import matplotlib.pyplot as plt


def normalize_image255(img):
    # Changes the input image range from (0, 255) to (0, 1)
    img = img.astype(np.float32)
    img = img/255.0
    return img


def normalize_image(img):
    # Normalizes the input image to range (0, 1) for visualization
    img = img.astype(np.float32)
    img = img - np.min(img)
    img = img/np.max(img)
    return img


# Get the paths to the training images
data_dir = 'scannn/data/input'
# print(os.listdir(data_dir))

# folderpaths = [os.path.join(data_dir, o) for o in os.listdir(
#     data_dir) if os.path.isdir(os.path.join(data_dir, o))]
# imagepaths = []
# print('folderpaths:', folderpaths)
# for folderpath in folderpaths:
#     temppaths = [os.path.join(folderpath, fname) for fname in os.listdir(
#         folderpath) if fname.endswith('.png')]
#     imagepaths += temppaths
# print('imagepaths:', imagepaths)

# Load and pre-process the training data
images = []
grayimages = []
filteredimages = []

# np.random.shuffle(imagepaths)
# for imagepath in imagepaths:
#     print('imagepath:', imagepath)
#     img = cv2.imread(imagepath).astype(np.float32)
#     img = normalize_image255(img)
# gray_img = make_grayscale(img)
# filtered_img = filter_image_sobelx(gray_img)

input_filenames = os.listdir(data_dir)
for fn in input_filenames:
    print('imagepath:', data_dir + fn)
    img = cv2.imread(data_dir + fn)
    img = normalize_image(img)

    images.append(img)
    grayimages.append(img)
    filteredimages.append(filtered_img)


input_height, input_width = img.shape


def linearcnn_model():
    # Returns a convolutional neural network model with a single linear convolution layer
    model = Sequential()
    model.add(Conv2D(1, (3, 3), padding='same',
                     input_shape=(input_height, input_width, 1)))
    return model


model = linearcnn_model()
sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

number_of_epochs = 100
loss = []
val_loss = []
convweights = []


for epoch in range(number_of_epochs):
    history_temp = model.fit(grayimages, filteredimages,
                             batch_size=4,
                             epochs=1,
                             validation_split=0.2)
    loss.append(history_temp.history['loss'][0])
    val_loss.append(history_temp.history['val_loss'][0])
    convweights.append(model.layers[0].get_weights()[0].squeeze())
