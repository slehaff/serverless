from PIL import Image
from PIL import ImageEnhance
import cv2


def brighter(folder):
    path = './testtf/data/'
    filename = path + folder + '/1nom.png'
    img = Image.open(filename)
    brighten = ImageEnhance.Brightness(img)
    img = brighten.enhance(3.0)
    img.save(path + folder + '/b1nom.png')
    filename = path + folder + '/1denom.png'
    img = Image.open(filename)
    brighten = ImageEnhance.Brightness(img)
    img = brighten.enhance(3.0)
    img.save(path + folder + '/b1denom.png')


# brighten('teeth1')
# brighten('teeth2')
# brighten('teeth3')
# brighten('teeth4')
# brighten('teeth5')
# brighten('teeth6')


def tweekshadow(input):
    for i in range(0, 480):
        for j in range(0, 640):
            if input[i, j] < 30:
                input[i, j] = 30
            if input[i,j] > 200:
                input[i, j] = 200
    return(input)


def equalizeImg(folder):
    path = './testtf/data/'
    filename = path + folder + '/b1nom.png'
    img = cv2.imread(filename, 0)
    equ = cv2.equalizeHist(img)
    equ = tweekshadow(equ)
    cv2.imwrite(path + folder + '/b1nom.png', equ)
    filename = path + folder + '/b1denom.png'
    img = cv2.imread(filename, 0)
    equ = cv2.equalizeHist(img)
    equ = tweekshadow(equ)
    cv2.imwrite(path + folder + '/b1denom.png', equ)


# equalizeImg('teeth1')
# equalizeImg('teeth2')
# equalizeImg('teeth3')
# equalizeImg('teeth4')
# equalizeImg('teeth5')
# equalizeImg('teeth6')
