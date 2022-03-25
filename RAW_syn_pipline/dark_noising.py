# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:12:03 2021

@author: hy
"""

import numpy as np
import scipy.stats as stats
from os.path import join
import random

class NoiseModel:
    def __init__(self, model='g', camera='CanonEOS5D4', cfa='bayer'):
        super().__init__()    
        self.camera = camera
        self.param_dir = './camera_params'

        print('[i] NoiseModel with {}'.format(self.param_dir))
        print('[i] camera: {}'.format(self.camera))
        print('[i] using noise model {}'.format(model))
        
        self.camera_params = {}
        self.camera_params[camera] = np.load(join(self.param_dir, camera+'_params.npy'), allow_pickle=True).item() 

        self.model = model
        
    def _sample_params(self):
        camera = self.camera
        Q_step = 1 

        profiles = ['Profile-1']
        saturation_level = 16383 - 512 

        
        camera_params = self.camera_params[camera]
        Kmin = camera_params['Kmin']
        Kmax = camera_params['Kmax']
        
        (camera_params['G_shape'])
        ind = np.random.randint(0, camera_params['color_bias'].shape[0])
        color_bias = camera_params['color_bias'][ind, :]
        profile = np.random.choice(profiles)
        camera_params = camera_params[profile]

        
        log_K = np.random.uniform(low=np.log(Kmin), high=np.log(Kmax))
        log_g_scale = np.random.standard_normal() * camera_params['g_scale']['sigma'] * 1 +\
             camera_params['g_scale']['slope'] * log_K + camera_params['g_scale']['bias']
        log_G_scale = np.random.standard_normal() * camera_params['G_scale']['sigma'] * 1 +\
             camera_params['G_scale']['slope'] * log_K + camera_params['G_scale']['bias']
        log_R_scale = np.random.standard_normal() * camera_params['R_scale']['sigma'] * 1 +\
             camera_params['R_scale']['slope'] * log_K + camera_params['R_scale']['bias']

        
        K = np.exp(log_K)
        g_scale = np.exp(log_g_scale)
        G_scale = np.exp(log_G_scale)
        R_scale = np.exp(log_R_scale)

        
        #ratio = np.random.uniform(low=100, high=300)
        ratio = np.random.uniform(low=100, high=200)
        # ratio = np.random.uniform(low=20, high=100)
        # ratio = np.random.uniform(low=1, high=300)
        # ratio = 1
        # ratio = np.random.uniform(low=20, high=50)
        return np.array([K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio]) 

    def __call__(self, y, params=None):
        if params is None:
            K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio = self._sample_params()
        else:
            K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio = params

        y = y * saturation_level
        y = y / ratio
        
        
        if 'P' in self.model:
            z = np.random.poisson(y / K).astype(np.float32) * K
        elif 'p' in self.model:
            z = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(K * y, 1e-10))
        else:
            z = y

        
        if 'g' in self.model:
            z = z + np.random.randn(*y.shape).astype(np.float32) * np.maximum(g_scale, 1e-10) # Gaussian noise            
        elif 'G' in self.model:
            z = z + stats.tukeylambda.rvs(G_shape, loc=0, scale=G_scale, size=y.shape).astype(np.float32) # Tukey Lambda 

        
        if 'B' in self.model:
            z = self.add_color_bias(z, color_bias=color_bias)

        
        if 'R' in self.model:
            z = self.add_banding_noise(z, scale=R_scale)

        
        if 'U' in self.model:
            z = z + np.random.uniform(low=-0.5*Q_step, high=0.5*Q_step)     

        z = z * ratio
        z = z / saturation_level

        return z

    def add_color_bias(self, img, color_bias): 
        channel = img.shape[2]
        img = img + color_bias.reshape((1, 1, channel))
        return img

    def add_banding_noise(self, img, scale):
        channel = img.shape[2]
        img = img + np.random.randn(img.shape[0], 1, channel).astype(np.float32) * scale
        return img

def adjust_random_brightness(image, s_range=(0.1, 0.3)): 
    assert s_range[0] < s_range[1]
    ratio = np.random.rand() * (s_range[1] - s_range[0]) + s_range[0]
    return image * ratio

def add_gaussian_noise(image, mean=0, std=0.25):
    noise = np.random.normal(mean, std, image.shape)
    return image + noise

def random_noise_levels(): 
  """Generates random noise levels from a log-log linear distribution."""
  log_min_shot_noise = np.log(0.0001)
  log_max_shot_noise = np.log(0.012)
  log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
  shot_noise = np.exp(log_shot_noise)

  line = lambda x: 2.18 * x + 1.20
  log_read_noise = line(log_shot_noise) + np.random.normal(0, 0.26) 
  read_noise = np.exp(log_read_noise)
  return shot_noise, read_noise

def add_read_and_shot_noise(image, shot_noise=0.01, read_noise=0.005): 
  """Adds random shot (proportional to image) and read (independent) noise."""
  variance = image * shot_noise + read_noise
  noise = np.random.normal(0, np.sqrt(variance), size=(variance.shape)) 
  return image + noise