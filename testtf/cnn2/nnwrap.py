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

def saveswat(i, nom, denom):
    folder = '/home/samir/serverless/'
    np.save(folder+ 'newnnnom/' + str(i) + '.npy', nom, allow_pickle=False)
    np.save(folder+ 'newnndenom/' + str(i) + '.npy', denom, allow_pickle=False)

def loadswat(i):
    folder ='/home/samir/serverless/' 
    nnnom = np.load(folder+ 'newnnnom/' + str(i) + '.npy')
    nndenom = np.load(folder+ 'newnndenom/' + str(i) + '.npy')
    return(nnnom, nndenom)

def testarctan(folder):
    nominator = folder + 'v22/b1nom.png'
    denominator = folder + 'v22/b1denom.png'
    test_im_wrap = nn2_wrap(nominator, denominator)
    png_file = folder + 'v22/npy_im_wrap.png'
    cv2.imwrite(png_file, test_im_wrap)


# folder = '/home/samir/serverless/testtf/data/' 
# testarctan(folder)


def nnfull_wrap(nom, denom):
    wrap = np.zeros((fullheight, fullwidth), dtype=np.float)
    im_wrap = np.zeros((fullheight, fullwidth), dtype=np.float)
    greynom = np.load(nom)
    greydenom = np.load(denom)
    for i in range(fullheight):
        for j in range(fullwidth):
            wrap[i, j] = np.arctan2(1.7320508 *greynom[i, j], greydenom[i, j])
            if wrap[i, j] < 0:
                if greynom[i, j] < 0:
                    wrap[i, j] += 2*np.pi
                else:
                    wrap[i, j] += 1 * np.pi
            im_wrap[i, j] = 128/np.pi * wrap[i, j]

    wrap = cv2.GaussianBlur(wrap, (3, 3), 0)
    im_wrap = cv2.GaussianBlur(im_wrap, (3, 3), 0)
    return(im_wrap)

def nnswat_wrap(nom, denom):
    wrap = np.zeros((rheight, rwidth), dtype=np.float)
    im_wrap = np.zeros((rheight, rwidth), dtype=np.float)
    greynom = np.load(nom)
    greydenom = np.load(denom)
    for i in range(rheight):
        for j in range(rwidth):
            wrap[i, j] = np.arctan2(1.7320508 *greynom[i, j], greydenom[i, j])
            if wrap[i, j] < 0:
                if greynom[i, j] < 0:
                    wrap[i, j] += 2*np.pi
                else:
                    wrap[i, j] += 1 * np.pi
            im_wrap[i, j] = 128/np.pi * wrap[i, j]

    wrap = cv2.GaussianBlur(wrap, (3, 3), 0)
    im_wrap = cv2.GaussianBlur(im_wrap, (3, 3), 0)
    return(im_wrap)

def save1nnwrap():
    for i in range(150):
        nnnom = '/home/samir/serverless/' +'newnnnom/' + str(i) +'.npy'
        nndenom = '/home/samir/serverless/' +'newnndenom/' + str(i) +'.npy'
        nnwrap = nnswat_wrap(nnnom, nndenom)
        pngfile = '/home/samir/serverless/' + 'nn1wrap/' + str(i) + '.png'
        cv2.imwrite(pngfile, nnwrap)


def save1wrap():
    for i in range(150):
        nnnom = '/home/samir/serverless/' +'newnom/' + str(i) +'.npy'
        nndenom = '/home/samir/serverless/' +'newdenom/' + str(i) +'.npy'
        nnwrap = nnswat_wrap(nnnom, nndenom)
        pngfile = '/home/samir/serverless/' + '1wrap/' + str(i) + '.png'
        cv2.imwrite(pngfile, nnwrap)


def testfullarctan(folder):
    nominator = folder + '1nom.npy'
    denominator = folder + '1denom.npy'
    test_im_wrap = nnfull_wrap(nominator, denominator)
    png_file = folder + 'npy_im_wrap.png'
    cv2.imwrite(png_file, test_im_wrap)

def testswatarctan(folder):
    nominator = folder + 'newnnnom/12.npy'
    denominator = folder + 'newnndenom/12.npy'
    test_im_wrap = nnswat_wrap(nominator, denominator)
    png_file = folder + 'npy_nn_im_wrap.png'
    cv2.imwrite(png_file, test_im_wrap)

# folder = '/home/samir/serverless/testtf/data/new_train/2scan_im_folder/' 
# testfullarctan(folder)
# greynom = np.load(folder + '1denom.npy')
# cv2.imwrite(folder + 'npy1denom.png',
#             (greynom).astype(np.uint8))
# greynom = np.load(folder + '1nom.npy')
# cv2.imwrite(folder + 'npy1nom.png',
#             (greynom).astype(np.uint8))

# folder = '/home/samir/serverless/' 
# testswatarctan(folder)
# greynom = np.load(folder + 'newnnnom/12.npy')
# greynom = 255.0*greynom
# cv2.imwrite(folder + 'npy0nom.png',
#             (greynom).astype(np.uint8))
# greynom = np.load(folder + 'newnndenom/12.npy')
# greynom = 255.0*greynom
# cv2.imwrite(folder + 'npy0denom.png',
#             (greynom).astype(np.uint8))