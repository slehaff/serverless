from PIL import Image
from enhance import *


indexarray= [[103,0],[271,10],[410,20],[103,137],[271,137],[410,137],[103,412], [271,271], [410,281]]


def imslice(imgpath, destfolder, offset, istart, iend, jstart, jend):
    img = Image.open(imgpath)
    n = offset
    out_height = 64
    out_width = 64
    for i in range(istart, iend):
        for j in range(jstart, jend):
            slice = img.crop((i*out_width, j*out_height, i *
                              out_width+64, j*out_height+64))
            slice.save(destfolder + str(n) + '.png')
            n += 1
    return

def im2slice(imgpath, destfolder, offset, winindex):
    img = Image.open(imgpath)
    n = offset
    for i in range(len(winindex)):
        slice = img.crop((winindex[i][0], winindex[i][1], winindex[i][0]+170, winindex[i][1]+170))
        slice.save(destfolder + str(n) + '.png')
        if i == 6:
            print(winindex[i][0], winindex[i][1], winindex[i][0]+170, winindex[i][1]+170)
        n+=1

    return



def folderLoad(offset, folder):
    imgpath1 = './testtf/data/' + folder + '/image3.png'
    destfolder1 = 'fringeA/'
    imgpath2 = './testtf/data/' + folder + '/gray.png'
    destfolder2 = 'gray/'
    imgpath3 = './testtf/data/' + folder + '/b1nom.png'
    destfolder3 = 'nom/'
    imgpath4 = './testtf/data/' + folder + '/b1denom.png'
    destfolder4 = 'denom/'
    istart = 2
    iend = 9
    jstart = 1
    jend = 6
    imslice(imgpath1, destfolder1, offset, istart, iend, jstart, jend)
    imslice(imgpath2, destfolder2, offset, istart, iend, jstart, jend)
    imslice(imgpath3, destfolder3, offset, istart, iend, jstart, jend)
    imslice(imgpath4, destfolder4, offset, istart, iend, jstart, jend)


def folder2load(offset, folder, winindex):
    imgpath1 = './testtf/data/' + folder + '/image3.png'
    destfolder1 = 'fringeA/'
    imgpath2 = './testtf/data/' + folder + '/gray.png'
    destfolder2 = 'gray/'
    imgpath3 = './testtf/data/' + folder + '/b1nom.png'
    destfolder3 = 'nom/'
    imgpath4 = './testtf/data/' + folder + '/b1denom.png'
    destfolder4 = 'denom/'
    im2slice(imgpath1, destfolder1, offset, winindex)
    im2slice(imgpath2, destfolder2, offset, winindex)
    im2slice(imgpath3, destfolder3, offset, winindex)
    im2slice(imgpath4, destfolder4, offset, winindex)




def makegray(folder):
    img = Image.open('./testtf/data/' + folder + '/image1.png').convert('L')
    img.save('./testtf/data/' + folder + '/gray.png')


# makegray('teeth1')
# makegray('teeth2')
# makegray('teeth3')
# makegray('teeth4')
# makegray('teeth5')
# makegray('teeth6')


# folderLoad(0, 'teeth1')
# folderLoad(35, 'teeth2')
# folderLoad(70, 'teeth3')
# folderLoad(105, 'teeth4')
# folderLoad(140, 'teeth5')
# folderLoad(175, 'teeth6')

def make(foldername, offset):
    brighten(foldername)
    equalizeImg(foldername)
    makegray(foldername)
    folder2load(offset, foldername, indexarray)


make('v21', 0)
make('v22', 9)
make('v23', 9*2)
make('v24', 9*3)
make('v25', 9*4)
make('v26', 9*5)
make('v27', 9*6)
make('v28', 9*7)
make('v29', 9*8)
make('v211', 9*9)
make('v212', 9*10)
make('v213', 9*11)
make('v214', 9*12)
make('v215', 9*13)
make('ttrain/1scan_im_folder', 9*15)
make('ttrain/2scan_im_folder', 9*16)
make('ttrain/3scan_im_folder', 9*17)
make('ttrain/4scan_im_folder', 9*18)
make('ttrain/5scan_im_folder', 9*19)
make('ttrain/6scan_im_folder', 9*20)
make('ttrain/7scan_im_folder', 9*21)
make('ttrain/8scan_im_folder', 9*22)
make('ttrain/9scan_im_folder', 9*23)
make('ttrain/10scan_im_folder', 9*24)
make('ttrain/11scan_im_folder', 9*25)
make('ttrain/12scan_im_folder', 9*26)
make('ttrain/13scan_im_folder', 9*27)
make('ttrain/14scan_im_folder', 9*28)
make('ttrain/15scan_im_folder', 9*29)
make('ttrain/16scan_im_folder', 9*30)
make('ttrain/17scan_im_folder', 9*31)
make('ttrain/18scan_im_folder', 9*32)
make('ttrain/19scan_im_folder', 9*33)
make('ttrain/20scan_im_folder', 9*34)
make('ttrain/21scan_im_folder', 9*35)
make('ttrain/22scan_im_folder', 9*36)
make('ttrain/23scan_im_folder', 9*37)
make('ttrain/24scan_im_folder', 9*38)


    
