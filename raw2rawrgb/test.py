import cv2
import numpy as np
import os

normal_output_folder = 'H:\\low_light\\process\\6720×4480_6type\\RAW_normal_6720×4480'
path=normal_output_folder+'\\1.png'



img=np.zeros((100,100,3)).astype(np.uint16)
cv2.imencode('.png',img)[1].tofile(path)