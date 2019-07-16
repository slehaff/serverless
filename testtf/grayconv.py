import cv2
import numpy as np


def make_grayscale(img):
    # Transform color image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


myfile = './testtf/data/teeth3_im_folder/image1.png'
img = cv2.imread(myfile).astype(np.float32)
# img = normalize_image255(img)
inp_img = make_grayscale(img)
cv2.imwrite('./testtf/data/teeth3_im_folder/gray_image.png', inp_img)
