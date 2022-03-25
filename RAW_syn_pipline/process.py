# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 01:14:23 2021

@author: hy
"""

import cv2
import numpy as np

def unpack_raw_bayer(img): 
    # unpack 4 channels to Bayer image
    img4c = np.transpose(img, (2, 0, 1))
    _, h, w = img4c.shape

    H = int(h * 2)
    W = int(w * 2)

    cfa_img = np.zeros((H, W), dtype=np.float32)

    cfa_img[0:H:2, 0:W:2] = img4c[0, :,:]
    cfa_img[0:H:2, 1:W:2] = img4c[1, :,:]
    cfa_img[1:H:2, 1:W:2] = img4c[2, :,:]
    cfa_img[1:H:2, 0:W:2] = img4c[3, :,:]
    
    return cfa_img
    
def process(img, shape):
    bayer = (unpack_raw_bayer(img) * 255.).round().astype(np.uint8) 
    RGB = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2RGB) 
    return cv2.resize(RGB, shape)
