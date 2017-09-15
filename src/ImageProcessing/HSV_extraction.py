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


if __name__ == "__main__":
	main()




















""" Code for producing subset_indices
for count in good_indices:
	temp = count % 10
	if 7 <= temp <=9:
		continue
	subset_indices.append(count)
np.save('subset70p_indices.npy', subset_indices)
"""







	# print (count)
	# current_image = path + str(count)
	# img = Image.open(current_image)
	# # img = misc.imread(current_image)
	# img = img.resize((128, 128), Image.ANTIALIAS) 
	# img.show()
	# # print (img.shape)
	# img = np.array(img)
	# if img.shape[2] != 3:
	# 	bad_indices.append(count)
	# 	good_indices.remove(count)
	# 	continue
	# arr = color.rgb2hsv(img)
	# IV_Current = arr[:,:,2]
	# IV.append(IV_Current)
	# IS_Current = arr[:,:,1]
	# IS.append(IV_Current)
	# IH_Current = arr[:,:,0]
	# IH.append(IV_Current)



# np.save('good_indices.npy', good_indices)
# np.save('bad_indices.npy', bad_indices)
# np.save('IV2.npy',IV)
# np.save('IS2.npy',IS)
# np.save('IH2.npy',IH)



# arr = color.rgb2hsv(misc.imread('img19'))
# IV = arr[:,:,0]

# print (IV.shape)
# print (IV[20][20])

# arr2 = misc.imread('img697')
# print (arr2.shape)
