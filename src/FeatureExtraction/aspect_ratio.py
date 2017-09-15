"""
This script goes through the image folder (for image indices in good_indices)
and saves the aspect ratio information in ../../Data/images_size.npy
"""
from PIL import Image
import numpy as np


def main():
    path = "../../data/photonet_dataset/images/"
    subset_indices = list(np.load('../../data/good_indices.npy'))
    image_sizes = []

    for index in subset_indices:
        current_image = path + str(index) + '.jpg'
        img = Image.open(current_image)
        img = np.array(img)
        image_sizes.append([img.shape[0], img.shape[1]])
<<<<<<< HEAD:Code/FeatureExtraction/aspect_ratio.py
    np.save('../../Data/image_sizes.npy', image_sizes)
=======
    np.save('../../data/image_sizes_40p.npy', image_sizes)
>>>>>>> 3c6c961f417c4a0cd6b7834fcaba9320faace3c8:src/FeatureExtraction/aspect_ratio.py


if __name__ == "__main__":
    main()
