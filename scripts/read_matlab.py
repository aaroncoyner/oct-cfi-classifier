import os

import scipy.io
import cv2
import numpy as np

def convert_mat_to_png(mat_f_path, png_f_path):
	try:
		mat = scipy.io.loadmat(mat_f_path)
		image_data = mat['data']

		if image_data.dtype != np.uint8:
			image_data = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

		cv2.imwrite(png_f_path, image_data)
	except:
		print('Error converting:' + mat_f_path)


DATA_PATH = '../data/segmentation-data/oct/matlab/'

for f_path in os.listdir(DATA_PATH):
	if not f_path.startswith('.'):
		png_f_path = '../data/segmentation-data/oct/oct-image/' + os.path.splitext(f_path)[0] + '.png'
		convert_mat_to_png(DATA_PATH + f_path, png_f_path)
		
