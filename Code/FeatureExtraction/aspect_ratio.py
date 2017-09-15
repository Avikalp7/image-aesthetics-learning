"""
This script goes through the image folder (for image indices in good_indices)
and saves the aspect ratio information in ../../Data/images_size.npy
"""
from PIL import Image
import numpy as np


def main():
    path = "../../Data/photonet_dataset/images/"
    subset_indices = list(np.load('../../Data/good_indices.npy'))
    image_sizes = []

    for index in subset_indices:
        current_image = path + str(index) + '.jpg'
        img = Image.open(current_image)
        img = np.array(img)
        image_sizes.append([img.shape[0], img.shape[1]])
    np.save('../../Data/image_sizes.npy', image_sizes)


if __name__ == "__main__":
    main()
