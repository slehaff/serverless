# Under development, will be takin g shots for calibration
import cv2
import numpy as np
import time


rwidth = 170
rheight = 170
fullwidth = 640
fullheight = 480

def sqdist(v1, v2):
    d = v1-v2
    return 1-d*d






def nn_wrap(nom, denom):
    wrap = np.zeros((rheight, rwidth), dtype=np.float)
    im_wrap = np.zeros((rheight, rwidth), dtype=np.float)
 
    for i in range(rheight):
        for j in range(rwidth):
            wrap[i, j] = np.arctan2(1.7320508 *nom[i, j], denom[i, j])
            if wrap[i, j] < 0:
                if nom[i, j] < 0:
                    wrap[i, j] += 2*np.pi
                else:
                    wrap[i, j] += 1 * np.pi
            im_wrap[i, j] = 128/np.pi * wrap[i, j]

    wrap = cv2.GaussianBlur(wrap, (3, 3), 0)
    im_wrap = cv2.GaussianBlur(im_wrap, (3, 3), 0)
    return(im_wrap)




def nn2_wrap(nom, denom):
    image = cv2.imread(nom)
    greynom = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(denom)
    greydenom = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    wrap = np.zeros((rheight, rwidth), dtype=np.float)
    im_wrap = np.zeros((rheight, rwidth), dtype=np.float)
 
    for i in range(rheight):
        for j in range(rwidth):
            wrap[i, j] = np.arctan2(1.7320508 *greynom[i, j], greydenom[i, j])
            if wrap[i, j] < 0:
                if nom[i, j] < 0:
                    wrap[i, j] += 2*np.pi
                else:
                    wrap[i, j] += 1 * np.pi
            im_wrap[i, j] = 128/np.pi * wrap[i, j]

    wrap = cv2.GaussianBlur(wrap, (3, 3), 0)
    im_wrap = cv2.GaussianBlur(im_wrap, (3, 3), 0)
    return(im_wrap)



def testarctan(folder):
    nominator = folder + 'v22/b1nom.png'
    denominator = folder + 'v22/b1denom.png'
    test_im_wrap = nn2_wrap(nominator, denominator)
    png_file = folder + 'b22_im_wrap.png'
    cv2.imwrite(png_file, test_im_wrap)


folder = '/home/samir/serverless/testtf/data/' 
testarctan(folder)
