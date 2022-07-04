# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:59:56 2022

@author: talha
"""

import random
from fmutils import fmutils as fmu
import cv2
import numpy as np
from tqdm import trange


def mosaic_augment(all_img_list, idxs, output_size, scale_range):
    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])

    for i, idx in enumerate(idxs):
        path = all_img_list[idx]
        

        img = cv2.imread(path)
        if i == 0:  # top-left
            img = cv2.resize(img, (divid_point_x, divid_point_y))
            output_img[:divid_point_y, :divid_point_x, :] = img
            

        elif i == 1:  # top-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, divid_point_y))
            output_img[:divid_point_y, divid_point_x:output_size[1], :] = img
            
        elif i == 2:  # bottom-left
            img = cv2.resize(img, (divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], :divid_point_x, :] = img
            
        else:  # bottom-right
            img = cv2.resize(img, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
            output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = img
            

    return output_img


img_dir = '../data/'
no_of_imgs = 40
OUTPUT_SIZE = (4000, 6000) 
SCALE_RANGE = (0.3, 0.7)

img_paths =  fmu.get_all_files(img_dir)

for i in trange(no_of_imgs):
    idxs = random.sample(range(len(img_paths)), 4)
    
    new_image = mosaic_augment(img_paths, idxs,
                               OUTPUT_SIZE, SCALE_RANGE)
    
    cv2.imwrite(f'../new/output_{i}.jpg', new_image)
    
