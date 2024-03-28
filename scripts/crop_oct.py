import os

from tqdm import tqdm
from PIL import Image

from PIL import Image

def center_square_crop_and_resize_image(image_path, output_path, new_size=(512, 512)):
    try:
        with Image.open(image_path) as img:
            square_size = img.height
            left = (img.width - square_size) / 2
            right = (img.width + square_size) / 2
            img_cropped = img.crop((left, 0, right, square_size))
            img_resized = img_cropped.resize(new_size)
            img_resized.save(output_path)
    except:
        print(f'Corrupt file {image_path}')
        os.remove(image_path)


folder = './data.nosync/oct/plus/'
for f_path in tqdm(os.listdir(folder)):
    if not f_path.startswith('.'):
        center_square_crop_and_resize_image(os.path.join(folder, f_path), os.path.join(folder, f_path))
