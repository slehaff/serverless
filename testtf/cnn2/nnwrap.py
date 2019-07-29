# Under development, will be takin g shots for calibration
import cv2
import numpy as np
import time


rwidth = 170
rheight = 170


def sqdist(v1, v2):
    d = v1-v2
    return 1-d*d




def get_A(im_arr):
    A = np.zeros((rwidth, rheight), dtype=np.float)
    for i in range(3):
        A = np.sum(A, im_arr[i])
    A = np.divide(A, 3)
    return A


def get_B(im_arr):
    for i in range(im_arr.length):
        B = np.multiply(im_arr[i], (np.sin(2*np.pi * i / im_arr.length)))

    return B


def get_average(array, n):
    b = np.add(array[0], array[1])
    average = 1/n * np.add(b, array[2])
    print('average size =', average.shape)
    return average



def nn_wrap(nom, denom):
    wrap = np.zeros((rheight, rwidth), dtype=np.float)
    im_wrap = np.zeros((rheight, rwidth), dtype=np.float)
 
    for i in range(rheight):
        for j in range(rwidth):
            wrap[i, j] = np.arctan2(1.7320508 * nom[i, j], denom[i, j])
            if wrap[i, j] < 0:
                if nom[i, j] < 0:
                    wrap[i, j] += 2*np.pi
                else:
                    wrap[i, j] += 1 * np.pi
            im_wrap[i, j] = 128/np.pi * wrap[i, j]

    # wrap = cv2.GaussianBlur(wrap, (3, 3), 0)
    # im_wrap = cv2.GaussianBlur(im_wrap, (3, 3), 0)
    return(im_wrap)
