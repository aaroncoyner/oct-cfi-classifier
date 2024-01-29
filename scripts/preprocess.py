import os
import cv2



def apply_clahe(src):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = cv2.imread(src + f_path, 0)
            img = clahe.apply(img)
            cv2.imwrite(src + f_path, img)


def resize_cfi(src, final_width=800, final_height=840):
    for f_path in os.listdir(src):
        if not f_path.startswith('.'):
            img = cv2.imread(src + f_path)
            img = cv2.copyMakeBorder(img, 180, 180, 80, 80, cv2.BORDER_CONSTANT,value=[0,0,0])
            cv2.imwrite(src + f_path, img)


apply_clahe('../data/segmentation-data/cfi/')
apply_clahe('../data/segmentation-data/oct/')

resize_cfi('../data/segmentation-data/cfi/')
resize_cfi('../data/segmentation-data/cfi-mask/')
