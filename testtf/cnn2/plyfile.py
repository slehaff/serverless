"""
plyfun
@author:wronk
Write surface to a .ply (Stanford 3D mesh file) in a way that preserves
vertex order in the MNE sense. Extendable for colors or other vertex/face properties.
.ply format: https://en.wikipedia.org/wiki/PLY_(file_format)
"""


# import mne
import numpy as np
from os import path as op
import os
from os import environ

import argparse
import sys
import os
from PIL import Image
import open3d as o3d




def write_surf2ply(rr, tris, save_path):
    out_file = open(save_path, 'w')

    head_strs = ['ply\n', 'format ascii 1.0\n']
    ele_1 = ['element vertex ' + str(len(rr)) + '\n',
             'property float x\n',
             'property float y\n',
             'property float z\n']
    ele_2 = ['element face ' + str(len(tris)) + '\n',
             'property list uchar int vertex_index\n']
    tail_strs = ['end_header\n']

    # Write Header
    out_file.writelines(head_strs)
    out_file.writelines(ele_1)
    out_file.writelines(ele_2)
    out_file.writelines(tail_strs)

    ##############
    # Write output
    ##############
    # First, write vertex positions
    for vert in rr:
        out_file.write(str(vert[0]) + ' ')
        out_file.write(str(vert[1]) + ' ')
        out_file.write(str(vert[2]) + '\n')

    # Second, write faces using vertex indices
    for face in tris:
        out_file.write(str(3) + ' ')
        out_file.write(str(face[0]) + ' ')
        out_file.write(str(face[1]) + ' ')
        out_file.write(str(face[2]) + '\n')

    out_file.close()


# if __name__ == '__main__':
#     struct_dir = op.join(environ['SUBJECTS_DIR'])
#     subject = 'AKCLEE_139'

#     surf_fname = op.join(struct_dir, subject, 'surf', 'lh.pial')
#     save_path = op.join('/media/Toshiba/Blender/Brains', subject,
#                         'lh.pial_reindex.ply')

#     rr, tris = mne.read_surface(surf_fname)
#     write_surf2ply(rr, tris, save_path)


#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# the resulting .ply file can be viewed for example with meshlab
# sudo apt-get install meshlab

"""
This script reads a registered pair of color and depth images and generates a
colored 3D point cloud in the PLY format.
"""


focalLength = 220.0
centerX = 320
centerY = 240
scalingFactor = 50.0

def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = Image.open(rgb_file)
    # depth = Image.open(depth_file)
    depth = Image.open(depth_file).convert('I')

    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = []    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            # Z = depth.getpixel((u,v)) / scalingFactor
            # if Z==0: continue
            # X = (u - centerX) * Z / focalLength
            # Y = (v - centerY) * Z / focalLength
            Z = depth.getpixel((u, v)) * .22
            if Z == 0: continue
            Y = .22 * v
            X = .22 * u
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='''
#     This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
#     PLY format. 
#     ''')
#     parser.add_argument('rgb_file', help='input color image (format: png)')
#     parser.add_argument('depth_file', help='input depth image (format: png)')
#     parser.add_argument('ply_file', help='output PLY file (format: ply)')
#     args = parser.parse_args()

#     generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)


# for i in range(150):
#     print(i)
#     rgbfile= 'new/colswt/' + str(i) + '.png'
#     depthfile = 'new/nnunwrap/' + str(i) + '.png'
#     plyfile= 'new/nnply/' + str(i) + '.ply'
#     generate_pointcloud(rgbfile, depthfile, plyfile)
#     rgbfile= 'new/colswt/' + str(i) + '.png'
#     depthfile = 'new/unwrap/' + str(i) + '.png'
#     plyfile= 'new/ply/' + str(i) + '.ply'
#     generate_pointcloud(rgbfile, depthfile, plyfile)
folder = '/home/samir/db2/scan/static/scan_folder/scan_im_folder/' 
rgbfile= folder + 'image1' + '.png'
depthfile = folder + 'unwrap' + '.png'
plyfile= folder + 'image3' + '.ply'
generate_pointcloud(rgbfile, depthfile, plyfile)