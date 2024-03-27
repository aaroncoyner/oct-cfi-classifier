import os

import cv2
import numpy as np
import scipy.io
from skimage.morphology import skeletonize
from skimage import io


def convert_matlab(f_path):
    mat = scipy.io.loadmat(f_path)
    img = mat['data']
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def pad_image(img, height=864, width=800):
    top_bottom = int((height - img.shape[0]) / 2)
    left_right = int((width - img.shape[1]) / 2)
    img = cv2.copyMakeBorder(img, top_bottom, top_bottom, left_right, left_right, cv2.BORDER_CONSTANT,value=[0,0,0])
    return img


def apply_clahe(img, clip_limit, tile_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    img = clahe.apply(img)
    return img


def process_oct(src, clip_limit=4.0, tile_size=(8,8), gaussian_kernel=(3,3)):
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = convert_matlab(src + f_path)
            if 'Image' in f_path:
                img = apply_clahe(img, clip_limit, tile_size)
                img = cv2.GaussianBlur(img, gaussian_kernel, 1)
                save_path = src + '../oct/' + os.path.splitext(f_path)[0].removesuffix('_Image') + '.png'
            else:
                save_path = src + '../oct-mask/' + os.path.splitext(f_path)[0].removesuffix('_Labels') + '.png'
            img = pad_image(img)
            cv2.imwrite(save_path, img)


def process_cfi(src, clip_limit=10.0, tile_size=(8,8), gaussian_kernel=(3,3),
                sharpening_kernel=np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]),
                noise_intensity=50):
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = cv2.imread(src + f_path)
            img = pad_image(img, height=840)
            img = img[:,:,1]
            img = cv2.filter2D(img, -1, sharpening_kernel)
            img = apply_clahe(img, clip_limit, tile_size)
            noise = np.random.randint(-noise_intensity, noise_intensity, img.shape, dtype='int16')
            img = cv2.add(img.astype('int16'), noise)
            img = np.clip(img, 0, 255).astype('uint8')
            img = cv2.GaussianBlur(img, gaussian_kernel, 1)
            img = pad_image(img)
            cv2.imwrite(src + f_path, img)


def process_segmentation(src):
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = io.imread(src + f_path, as_gray=True)
            img = pad_image(img)
            img = (img > 0.5).astype(np.uint8)
            img = skeletonize(img).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            img = cv2.dilate(img, kernel, iterations=1)
            cv2.imwrite(src + f_path, img)

os.makedirs('../data/segmentation-data/oct/')
os.makedirs('../data/segmentation-data/oct-mask/')

process_oct('../data/segmentation/matlab/')
process_segmentation('../data/segmentation/oct-mask/')

process_cfi('../data/segmentation/cfi/')
process_segmentation('../data/segmentation/cfi-mask/')
