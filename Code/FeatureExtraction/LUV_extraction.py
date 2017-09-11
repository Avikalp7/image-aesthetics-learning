"""
Code to go through the image dataset and save LUV matrix associated
"""
from scipy import misc
import numpy as np
from skimage import color
from skimage import data
import os
import PIL
from PIL import Image


path = "/home/avikalp/semester6/BS/twitter_images/img"

good_indices = list(np.load('good_indices.npy'))
bad_indices = []
IV = []
IU = []
IL = []
LUV = []
subset_indices = np.load('subset40p_indices.npy')



for count in subset_indices:
	print (count)
	current_image = path + str(count)
	img = Image.open(current_image)
	# img = misc.imread(current_image)
	img = img.resize((128, 128), Image.ANTIALIAS) 
	# img.show()
	# print (img.shape)
	img = np.array(img)
	# if img.shape[2] != 3:
	# 	bad_indices.append(count)
	# 	good_indices.remove(count)
	# 	continue
	arr = color.rgb2luv(img)
	LUV.append(arr)
	# IV_Current = arr[:,:,2]
	# IV.append(IV_Current)
	# IU_Current = arr[:,:,1]
	# IU.append(IU_Current)
	# IL_Current = arr[:,:,0]
	# IL.append(IL_Current)


# # np.save('good_indices.npy', good_indices)
# # np.save('bad_indices.npy', bad_indices)
# np.save('LUV_V_40p.npy',IV)
# np.save('LUV_U_40p.npy',IU)
# np.save('LUV_L_40p.npy',IL)
np.save('LUV_40p.npy', LUV)