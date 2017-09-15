from __future__ import division

from scipy import misc
from skimage import color
from skimage import data
from PIL import Image
from pywt import wavedec2
from sklearn.cluster import KMeans
import PIL
import numpy as np

import os


global IH, IS, IV, path, image_sizes
global LH, HL, HH, S1, S2, S3
global _f10, _f11, _f12, _f13, _f14, _f15, _f16, _f17, _f18


# Some images (b/w) give zero values on S1, S2, S3 - leading to division by zero
def check_zero(epsilon = 50):
	global S1, S2, S3
	if S1 == 0:
		S1 = epsilon
	if S2 == 0:
		S2 = epsilon
	if S3 == 0:
		S3 = epsilon


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f10_f11_f12(i):
	global S1, S2, S3, LH, HL, HH
	coeffs = wavedec2(IH[i], 'db1', level = 3)
	LL, LH, HL, HH = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f13_f14_f15(i):
	global S1, S2, S3, LL, HL, HH
	coeffs = wavedec2(IS[i], 'db1', level = 3)
	LL, LH, HL, HH = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f16_f17_f18(i):
	global S1, S2, S3, LL, HL, HH
	coeffs = wavedec2(IV[i], 'db1', level = 3)
	LL, LH, HL, HH = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()


# Exposure of Light
def f1(i):
	return sum(sum(IV[i]))/(IV.shape[0] * IV.shape[1])


# Average Saturation / Saturation Indicator
def f3(i):
	return sum(sum(IS[i]))/(IS.shape[0] * IS.shape[1])	


# Average Hue / Hue Indicator
def f4(i):
	return sum(sum(IH[i]))/(IH.shape[0] * IH.shape[1])


# Average hue in inner rectangle for rule of thirds inference
def f5(i):
	X = IH[i].shape[0]
	Y = IH[i].shape[1]
	return sum(sum(IH[i, X/3 : 2*X/3, Y/3 : 2*Y/3])) * 9 / (X * Y)


# Average saturation in inner rectangle for rule of thirds inference
def f6(i):
	X = IS[i].shape[0]
	Y = IS[i].shape[1]
	return sum(sum(IS[i, X/3 : 2*X/3, Y/3 : 2*Y/3])) * (9/(X * Y))


# Average V in inner rectangle for rule of thirds inference
def f7(i):
	X = IV[i].shape[0]
	Y = IV[i].shape[1]
	return sum(sum(IV[i, X/3 : 2*X/3, Y/3 : 2*Y/3])) * (9/(X * Y))


# Spacial Smoothness of first level of Hue property
def f10(i):
	global _f10
	prereq_f10_f11_f12(i)
	_f10 = (1/S1)*(sum(sum(HH[0])) + sum(sum(HL[0])) + sum(sum(LH[0])))
	return _f10
	

# Spacial Smoothness of second level of Hue property
def f11(i):
	global _f11
	_f11 = (1/S2)*(sum(sum(HH[1])) + sum(sum(HL[1])) + sum(sum(LH[1])))
	return _f11


# Spacial Smoothness of third level of Hue property
def f12(i):
	global _f12
	_f12 = (1/S3)*(sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2])))
	return _f12


# Spacial Smoothness of first level of Saturation property
def f13(i):
	global _f13
	prereq_f13_f14_f15(i)
	_f13 = (1/S1)*(sum(sum(HH[0])) + sum(sum(HL[0])) + sum(sum(LH[0])))
	return _f13

# Spacial Smoothness of second level of Saturation property
def f14(i):
	global _f14
	_f14 = (1/S2)*(sum(sum(HH[1])) + sum(sum(HL[1])) + sum(sum(LH[1])))
	return _f14


# Spacial Smoothness of third level of Saturation property
def f15(i):
	global _f15
	_f15 = (1/S3)*(sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2])))
	return _f15


# Spacial Smoothness of first level of Intensity property
def f16(i):
	global _f16
	prereq_f16_f17_f18(i)
	_f16 = (1/S1)*(sum(sum(HH[0])) + sum(sum(HL[0])) + sum(sum(LH[0])))
	return _f16


# Spacial Smoothness of second level of Intensity property
def f17(i):
	global _f17
	_f17 = (1/S2)*(sum(sum(HH[1])) + sum(sum(HL[1])) + sum(sum(LH[1])))
	return _f17


# Spacial Smoothness of third level of Intensity property
def f18(i):
	global _f18
	_f18 = (1/S3)*(sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2])))
	return _f18


# Sum of the average wavelet coefficients over all three frequency levels of Hue property
def f19(i):
	return _f10 + _f11 + _f12


# Sum of the average wavelet coefficients over all three frequency levels of Saturation property
def f20(i):
	return _f13 + _f14 + _f15


# Sum of the average wavelet coefficients over all three frequency levels of Intensity property
def f21(i):
	return _f16 + _f17 + _f18


# Image Size feature
def f22(i):
	return image_sizes[i][0] + image_sizes[i][1]


# Aspect Ratio Feature
def f23(i):
	return image_sizes[i][0] / float(image_sizes[i][1])	


# DoF feature for Hue property
def f53(i):
	prereq_f10_f11_f12(i)
	numerator = sum(sum(HH[2][32:95][32:95])) + sum(sum(HL[2][32:95][32:95])) + sum(sum(LH[2][32:95][32:95])) 
	denominator = sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2]))
	return numerator/denominator


# DoF feature for Saturation property
def f54(i):
	prereq_f13_f14_f15(i)
	numerator = sum(sum(HH[2][32:95][32:95])) + sum(sum(HL[2][32:95][32:95])) + sum(sum(LH[2][32:95][32:95]))
	denominator = sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2]))
	return numerator/denominator


# DoF feature for Intensity property
def f55(i):
	prereq_f16_f17_f18(i)
	numerator = sum(sum(HH[2][32:95][32:95])) + sum(sum(HL[2][32:95][32:95])) + sum(sum(LH[2][32:95][32:95]))
	denominator = sum(sum(HH[2])) + sum(sum(HL[2])) + sum(sum(LH[2]))
	return numerator/denominator





path = "/home/avikalp/semester6/BS/twitter_images/img"

if __name__ == '__main__':
	subset_indices = list(np.load('subset40p_indices.npy'))
	image_sizes = list(np.load('image_sizes_40p.npy'))

	IH = np.load('IH_40p.npy')
	IS = np.load('IS_40p.npy')
	IV = np.load('IV_40p.npy')

	feature_vec = []
	for i, index in enumerate(subset_indices):
		print (i)
		feature_vec.append([])
		feature_vec[i].append(f1(i))
		# feature_vec[i].append(f2(i))
		feature_vec[i].append(f3(i))
		feature_vec[i].append(f4(i))
		feature_vec[i].append(f5(i))
		feature_vec[i].append(f6(i))
		feature_vec[i].append(f7(i))
		# feature_vec[i].append(f8(i))
		# feature_vec[i].append(f9(i))
		feature_vec[i].append(f10(i))
		feature_vec[i].append(f11(i))
		feature_vec[i].append(f12(i))
		feature_vec[i].append(f13(i))
		feature_vec[i].append(f14(i))
		feature_vec[i].append(f15(i))
		feature_vec[i].append(f16(i))
		feature_vec[i].append(f17(i))
		feature_vec[i].append(f18(i))
		feature_vec[i].append(f19(i))
		feature_vec[i].append(f20(i))
		feature_vec[i].append(f21(i))
		feature_vec[i].append(f22(i))
		feature_vec[i].append(f23(i))
		# feature_vec[i].append(f24(i))
		# feature_vec[i].append(f25(i))
		# feature_vec[i].append(f26(i))
		# feature_vec[i].append(f27(i))
		# feature_vec[i].append(f28(i))
		# feature_vec[i].append(f29(i))
		# feature_vec[i].append(f30(i))
		# feature_vec[i].append(f31(i))
		# feature_vec[i].append(f32(i))
		# feature_vec[i].append(f33(i))
		# feature_vec[i].append(f34(i))
		# feature_vec[i].append(f35(i))
		# feature_vec[i].append(f36(i))
		# feature_vec[i].append(f37(i))
		# feature_vec[i].append(f38(i))
		# feature_vec[i].append(f39(i))
		# feature_vec[i].append(f40(i))
		# feature_vec[i].append(f41(i))
		# feature_vec[i].append(f42(i))
		# feature_vec[i].append(f43(i))
		# feature_vec[i].append(f44(i))
		# feature_vec[i].append(f45(i))
		# feature_vec[i].append(f46(i))
		# feature_vec[i].append(f47(i))
		# feature_vec[i].append(f48(i))
		# feature_vec[i].append(f49(i))
		# feature_vec[i].append(f50(i))
		# feature_vec[i].append(f51(i))
		# feature_vec[i].append(f52(i))
		feature_vec[i].append(f53(i))
		feature_vec[i].append(f54(i))
		feature_vec[i].append(f55(i))
		# feature_vec[i].append(f56(i))

		# -------------------------- #

		# Do something if required
		#
		#

		# del feature_vec[i][:]

np.save('feature_vecs.npy', feature_vec)



