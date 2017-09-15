"""
Code to go through the image dataset and save H,S,V matrices associated
"""

from scipy import misc
from PIL import Image
from skimage import color
from skimage import data
import numpy as np
import PIL

import os

# Change this to the path of a folder containing images with naming convention: "img<num>.png", eg "img442.jpg"
path = "/home/avikalp/semester6/SIGIR/twitter_images/img"


def main():
	# Load indices for images that were successfully downloaded
	good_indices = list(np.load('good_indices.npy'))

	IV = []
	IS = []
	IH = []

	# Use this to extract values for a sample 40% of good indices
	# To use all indices, set subset_indices to good_indices
	subset_indices = np.load('..subset40p_indices.npy')


	for count in subset_indices:
		print (count)
		current_image = path + str(count)
		img = Image.open(current_image)
		# img = misc.imread(current_image)
		img = img.resize((128, 128), Image.ANTIALIAS) 
		img.show()
		img = np.array(img)
		arr = color.rgb2hsv(img)
		IV_Current = arr[:,:,2]
		IV.append(IV_Current)
		IS_Current = arr[:,:,1]
		IS.append(IS_Current)
		IH_Current = arr[:,:,0]
		IH.append(IH_Current)


	np.save('../../data/IV_40p.npy',IV)
	np.save('../../data/IS_40p.npy',IS)
	np.save('../../data/IH_40p.npy',IH)
	return


if __name__ == "__main__":
	main()
