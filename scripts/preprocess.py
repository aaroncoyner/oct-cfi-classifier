import os
import cv2
import scipy.io
import numpy as np



def convert_matlab(src, dest):
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            mat = scipy.io.loadmat(src + f_path)
            image_data = mat['data']

            if image_data.dtype != np.uint8:
                image_data = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if 'Image' in f_path:
                save_path = dest + 'oct/' + os.path.splitext(f_path)[0] + '.png'
            else:
                save_path = dest + 'oct-mask/' + os.path.splitext(f_path)[0] + '.png'

            cv2.imwrite(save_path, image_data)


def pad_images(src):
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = cv2.imread(src + f_path)
            top_bottom = int((864 - img.shape[0]) / 2)
            left_right = int((800 - img.shape[1]) / 2)
            img = cv2.copyMakeBorder(img, top_bottom, top_bottom, left_right, left_right, cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imwrite(src + f_path, img)


def apply_clahe(src):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = cv2.imread(src + f_path, 0)
            img = clahe.apply(img)
            cv2.imwrite(src + f_path, img)


convert_matlab('../data/segmentation-data/matlab/', '../data/segmentation-data/')

pad_images('../data/segmentation-data/oct/')
pad_images('../data/segmentation-data/oct-mask/')
pad_images('../data/segmentation-data/cfi/')
pad_images('../data/segmentation-data/cfi-mask/')

apply_clahe('../data/segmentation-data/cfi/')
apply_clahe('../data/segmentation-data/oct/')
