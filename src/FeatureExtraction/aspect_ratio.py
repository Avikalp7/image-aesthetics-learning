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
    np.save('../../data/image_sizes_40p.npy', image_sizes)


if __name__ == "__main__":
    main()
