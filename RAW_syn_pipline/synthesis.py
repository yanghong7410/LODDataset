# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:30:15 2021

@author: hy
"""

import os
import os.path as osp

import numpy as np
from PIL import Image

from process import process
from unprocess import unprocess
from dark_noising import *


def main(input_folder, output_folder, noise_type):
    if not osp.exists(output_folder):
        os.mkdir(output_folder)

    if noise_type == 'gaussian':
        noisemodel = NoiseModel(model='g', camera='CanonEOS5D4')  
    elif noise_type == 'gaussian-poisson':
        noisemodel = NoiseModel(model='pg', camera='CanonEOS5D4')  
    elif noise_type == 'physics-based':
        noisemodel = NoiseModel(model='PGBRU', camera='CanonEOS5D4')  

    for filename in sorted(os.listdir(input_folder)):
        image = Image.open(osp.join(input_folder, filename))  
        if image.mode != 'RGB':
            image = image.convert("RGB")
        W, H = image.size

        # resize for mosaic in unprocessing
        image = image.resize((W // 2 * 2, H // 2 * 2)) 

        image = np.array(image).astype(np.float32) / 255.  
        raw, metadata = unprocess(image)  

        if noise_type is None:
            # dark_raw = adjust_random_brightness(raw, s_range=(0.2, 0.4)) 
            dark_raw = raw

        else:
            noisy_raw = noisemodel(raw)  
            noisy_raw = np.clip(noisy_raw, 0, 1)  
            # dark_raw = adjust_random_brightness(noisy_raw, s_range=(0.2, 0.4)) 
            dark_raw = noisy_raw

        result = process(dark_raw, (W, H))  
        result = Image.fromarray(result)
        result.save(osp.join(output_folder, filename[:-4] + '.png'))  


if __name__ == "__main__":
    main(input_folder=r'G:\coco2017\COCO2017\train2017',  
         output_folder=r'G:\object_detection_data\supp\coco+gaussian',
         noise_type='gaussian')  # None, 'gaussian', 'gaussian-poisson', 'physics-based'
