# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 23:28:43 2021

@author: hy
"""

import os
import os.path as osp

import rawpy
from tqdm import tqdm
import exifread

import numpy as np
from PIL import Image
import cv2

def extract_exposure(raw_path): 
    raw_file = open(raw_path, 'rb')
    exif_file = exifread.process_file(raw_file, details=False, strict=True)

    if 'EXIF ExposureTime' in exif_file:
        exposure_str = exif_file['EXIF ExposureTime'].printable
    else:
        exposure_str = exif_file['Image ExposureTime'].printable
    if '/' in exposure_str:
        fenmu = float(exposure_str.split('/')[0])
        fenzi = float(exposure_str.split('/')[-1])
        exposure = fenmu / fenzi
    else:
        exposure = float(exposure_str)
    return exposure

def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float16)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float16)

    white_point = raw.white_level
    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float16)
    # print(black_level[0,0,0], white_point)
    
    out = np.maximum(out - black_level,0) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out

def postprocess_bayer(rawpath, img4c):    
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)

    white_point = raw.white_level
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    img4c = np.minimum(img4c * (white_point - black_level) + black_level,white_point)

    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=False, no_auto_bright=True, output_bps=16, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    # out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None, gamma=(2.2, 0))

    # out = raw.postprocess(user_wb=[1,1,1,1], half_size=True, output_bps=8, no_auto_bright=True)
    return out



normal_input_folder = r'C:\Users\LANCE_hy\Desktop\normal'
dark_input_folder = r'C:\Users\LANCE_hy\Desktop\dark'


normal_output_folder = r'C:\Users\LANCE_hy\Desktop\1'
dark_output_folder = r'C:\Users\LANCE_hy\Desktop\2'

if not osp.exists(normal_output_folder):
    os.mkdir(normal_output_folder)
if not osp.exists(dark_output_folder):
    os.mkdir(dark_output_folder)

for filename in tqdm(sorted(os.listdir(normal_input_folder))):
    if filename[-3:] == 'CR2': 
        # prefix = filename[:4] 
        # number = filename[4:-4] 
        # suffix = filename[-4:] 
        # print(prefix, number, suffix)
        # dark_name = prefix + str(int(number) + 1) + suffix 
        dark_name = str(int(filename[:-4]) + 1) + '.CR2'
        # print(dark_name)
        if osp.exists(osp.join(dark_input_folder, dark_name)): 
            normal_raw = rawpy.imread(osp.join(normal_input_folder, filename)) 
            normal_img4c = pack_raw_bayer(normal_raw) 
            normal_img = postprocess_bayer(osp.join(normal_input_folder, filename), normal_img4c)
            normal_img = cv2.resize(normal_img, (6720, 4480)) 
            normal_img=cv2.cvtColor(normal_img,cv2.COLOR_RGB2BGR)
            print(osp.join(normal_output_folder, filename[:-4]+'.png'))
            # cv2.imwrite(osp.join(normal_output_folder, filename[:-4]+'.png'),normal_img)
            cv2.imencode('.png', normal_img)[1].tofile(osp.join(normal_output_folder, filename[:-4]+'.png'))

            # normal_img = Image.fromarray(normal_img) 
            # normal_img.save(osp.join(normal_output_folder, filename[:-4]+'.png')) 

            
            normal_exposure = extract_exposure(osp.join(normal_input_folder, filename)) 
            dark_exposure = extract_exposure(osp.join(dark_input_folder, dark_name)) 

            dark_raw = rawpy.imread(osp.join(dark_input_folder, dark_name)) 
            dark_img4c = pack_raw_bayer(dark_raw) / dark_exposure * normal_exposure 
            dark_img = postprocess_bayer(osp.join(dark_input_folder, dark_name), dark_img4c) 
            dark_img = cv2.resize(dark_img, (6720, 4480)) 
            dark_img = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(osp.join(dark_output_folder, dark_name[:-4]+'.png'),dark_img)
            cv2.imencode('.png', dark_img)[1].tofile(osp.join(dark_output_folder, dark_name[:-4]+'.png'))
            # dark_img = Image.fromarray(dark_img) 
            # dark_img.save(osp.join(dark_output_folder, dark_name[:-4]+'.png')) 