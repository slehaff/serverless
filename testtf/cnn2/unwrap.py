import numpy as np
import cv2
import argparse
import sys
import os
from PIL import Image
from jsoncloud import *


high_freq = 10
rwidth = 170
rheight = 170



def unwrap_r(low_f_file, high_f_file, destination):
    file1 = low_f_file
    file2 = high_f_file
    print("file1:", file1)
    print("file2:", file2)
    wrap1data = np.zeros((rheight, rwidth), dtype=np.float)
    wrap2data = np.zeros((rheight, rwidth), dtype=np.float)
    wrap1data = np.load(file1)  # To be continued
    wrap2data = np.load(file2)
    unwrapdata = np.zeros((rheight, rwidth), dtype=np.float)
    kdata = np.zeros((rheight, rwidth), dtype=np.float)
    # wrap1data = cv2.GaussianBlur(wrap1data, (0, 0), 3, 3)
    # wrap2data = cv2.GaussianBlur(wrap2data, (0, 0), 4, 4)
    for i in range(rheight):
        for j in range(rwidth):
            kdata[i, j] = round(
                (high_freq * 1.0 * wrap1data[i, j] - 1.0 * wrap2data[i, j])/2.0)
            unwrapdata[i, j] = (1.0 * wrap2data[i, j] +
                                2.0*kdata[i, j]*np.pi)/2

    print("I'm in nn unwrap_r")
    wr_save = destination + '.npy'
    np.save(wr_save, unwrapdata, allow_pickle=False)
    print(wr_save)
    unwrapdata = np.multiply(unwrapdata, 2.0)
    cv2.imwrite(destination + '.png', unwrapdata)


# def deduct_ref(unwrap, reference, folder1, folder2):
#     file1 = folder1 + '/' + unwrap
#     file2 = folder2 + '/' + reference
#     wrap_data = np.load(file1)  # To be continued
#     ref_data = np.load(file2)
#     net_data = np.subtract(ref_data, wrap_data)
#     net_save = folder1 + 'net_wrap.npy'
#     np.save(net_save, net_data, allow_pickle=False)
#     print(net_save)
#     # np.save('wrap24.pickle', wrap24data, allow_pickle=True)
#     net_data = np.multiply(net_data, 1.0)
#     cv2.imwrite(folder1 + 'abs_unwrap.png', net_data)


def unwrap(request):
    # folder = ScanFolder.objects.last().folderName
    folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/'
    ref_folder = '/home/samir/db2/scan/static/scan_folder/scan_ref_folder'
    three_folder = '/home/samir/db2/3D/static/3scan_folder'
    unwrap_r('scan_wrap2.npy', 'scan_wrap1.npy', folder )
    generate_json_pointcloud(folder + 'image1.png', folder +
                             '/unwrap.png', three_folder + '/pointcl.json')
    return render(request, 'scantemplate.html')



for i in range(150):
    flow = 'new/4/nnwrap/' + str(i) +'.npy'
    fhigh = 'new/1/nnwrap/' + str(i) + '.npy'
    destination = 'new/nnunwrap/' + str(i)
    unwrap_r(flow, fhigh, destination)
    rgb = 'new/colswt/' + str(i) + '.png'
    depth = 'new/nnunwrap/'+ str(i) + '.png'
    pcl = 'new/nn3djson/'+ str(i) + '.json'
    generate_json_pointcloud(rgb, depth, pcl)
    flow = 'new/4/wrap/' + str(i) +'.npy'
    fhigh = 'new/1/wrap/' + str(i) + '.npy'
    destination = 'new/unwrap/' + str(i)
    unwrap_r(flow, fhigh, destination)
    rgb = 'new/colswt/' + str(i) + '.png'
    depth = 'new/unwrap/'+ str(i) + '.png'
    pcl = 'new/3djson/'+ str(i) + '.json'
    generate_json_pointcloud(rgb, depth, pcl)