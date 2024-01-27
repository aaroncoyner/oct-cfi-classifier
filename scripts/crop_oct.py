import os

from tqdm import tqdm
from PIL import Image

from PIL import Image

def center_square_crop_and_resize_image(image_path, output_path, new_size=(512, 512)):
    # Open the image file
    try:
        with Image.open(image_path) as img:
            # Determine the size for the square crop (height of the image)
            square_size = img.height

            # Calculate the left and right coordinates for the crop
            left = (img.width - square_size) / 2
            right = (img.width + square_size) / 2

            # Perform the crop
            img_cropped = img.crop((left, 0, right, square_size))

            # Resize the image
            img_resized = img_cropped.resize(new_size)

            # Save the resized image
            img_resized.save(output_path)
    except:
        print(f'Corrupt file {image_path}')
        os.remove(image_path)


folder = './data.nosync/oct/plus/'
for f_path in tqdm(os.listdir(folder)):
    if not f_path.startswith('.'):
        center_square_crop_and_resize_image(os.path.join(folder, f_path), os.path.join(folder, f_path))
