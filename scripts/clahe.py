import cv2
import matplotlib.pyplot as plt

def apply_clahe(image_path, clip_limit, tile_grid_size):
    # Read the image
    img = cv2.imread(image_path, 0)  # 0 flag reads image in grayscale

    # Create a CLAHE object with specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img)

    # Display the images
    plt.figure(figsize=(12, 6))

    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(clahe_img, cmap='gray'), plt.title('CLAHE Image')
    plt.xticks([]), plt.yticks([])

    plt.show()

# Example usage
image_path = '/Users/coyner/Desktop/side-projects/adam.nosync/arvo-project/data.nosync/train_combined/normal/7256.png'
clip_limit = 2.0  # Adjust this value as needed
tile_grid_size = (4,4)  # Adjust this value as needed (in width, height)

apply_clahe(image_path, clip_limit, tile_grid_size)
