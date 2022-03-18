import os
import cv2

raw_dir = "/cfs/DataSets/raw_did/test_data/test_real_denoised/"
raw_images_list = os.listdir(raw_dir)
save_dir = "/cfs/DataSets/raw_did/test_data/test_real_denoised_512_512/"

for idx,name in enumerate(raw_images_list):
    img = cv2.imread(raw_dir + name)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite(save_dir + name, img)
    print(idx)



