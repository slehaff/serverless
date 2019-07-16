from PIL import Image


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


def folderLoad(offset, folder,):
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


def makegray(folder):
    img = Image.open('./testtf/data/' + folder + '/image1.png').convert('L')
    img.save('./testtf/data/' + folder + '/gray.png')


makegray('teeth1')
makegray('teeth2')
makegray('teeth3')
makegray('teeth4')
makegray('teeth5')
makegray('teeth6')


folderLoad(0, 'teeth1')
folderLoad(35, 'teeth2')
folderLoad(70, 'teeth3')
folderLoad(105, 'teeth4')
folderLoad(140, 'teeth5')
folderLoad(175, 'teeth6')
