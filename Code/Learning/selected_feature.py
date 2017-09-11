from __future__ import division
from scipy import misc
import numpy as np
from skimage import color
from skimage import data
import os
import PIL
from PIL import Image
from pywt import wavedec2
from sklearn.cluster import KMeans

from disjoint_sets import Graph
# from disjoint_sets import countIslands

global IH, IS, IV, path, image_sizes
global LH, HL, HH, S1, S2, S3
global _f10, _f11, _f12, _f13, _f14, _f15, _f16, _f17, _f18

# Parameter K for Kmeans is set here
kmeans_cluster_num = 12

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
	HL = LH = HH = [0]*3
	coeffs = wavedec2(IH[i], 'db1', level = 3)
	LL, (HL[2], LH[2], HH[2]), (HL[1], LH[1], HH[1]), (HL[0], LH[0], HH[0]) = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	# print('S1, S2, S3',S1, S2, S3)
	check_zero()


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f13_f14_f15(i):
	global S1, S2, S3, LL, HL, HH
	HL = LH = HH = [0]*3
	coeffs = wavedec2(IS[i], 'db1', level = 3)
	LL, (HL[2], HL[2], HH[2]), (HL[1], HL[1], HH[1]), (HL[0], HL[0], HH[0]) = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()


# Prerequiste for features _f10,11,12, calculating LL, LH, HL, HH for 3-level 2-D Discrete Wavelet Transform
def prereq_f16_f17_f18(i):
	global S1, S2, S3, LL, HL, HH
	HL = LH = HH = [0]*3
	coeffs = wavedec2(IV[i], 'db1', level = 3)
	LL, (HL[2], HL[2], HH[2]), (HL[1], HL[1], HH[1]), (HL[0], HL[0], HH[0]) = coeffs
	S1 = sum(sum(abs(LH[0]))) + sum(sum(abs(HL[0]))) + sum(sum(abs(HH[0])))
	S2 = sum(sum(abs(LH[1]))) + sum(sum(abs(HL[1]))) + sum(sum(abs(HH[1])))
	S3 = sum(sum(abs(LH[2]))) + sum(sum(abs(HL[2]))) + sum(sum(abs(HH[2]))) 
	check_zero()


def segmentation(graph):
	row = len(graph)
	col = len(graph[0])
	g = Graph(row, col, graph)
	dic = {}
	for cluster_num in range(kmeans_cluster_num):
		# print ("Number of points in cluster number", cluster_num, "is: ")
		dic[cluster_num] = g.countIslands(cluster_num)
		# print('Len pathces = ', len(dic[cluster_num][1]), ' Len lis = ', len(dic[cluster_num][0]))
	# print('i, BLOB_COUNT = ', i, blob_count)
	# print('Ending K-Means')

	return dic


def segments(dic):
	all_lengths = []
	all_patches = []
	for key in dic:
		all_lengths += dic[key][0]
		all_patches += dic[key][1]
	# print (len(all_lengths), len(all_patches))
	all_lengths = np.array(all_lengths)
	all_patches = np.array(all_patches)
	max_5_indices = all_lengths.argsort()[-5:][::-1]	# np.array
	return all_patches[max_5_indices]


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
	return sum(sum(IH[i, int(X/3) : int(2*X/3), int(Y/3) : int(2*Y/3)])) * 9 / (X * Y)


# Average saturation in inner rectangle for rule of thirds inference
def f6(i):
	X = IS[i].shape[0]
	Y = IS[i].shape[1]
	return sum(sum(IS[i, int(X/3) : int(2*X/3), int(Y/3) : int(2*Y/3)])) * (9/(X * Y))


# Average V in inner rectangle for rule of thirds inference
def f7(i):
	X = IV[i].shape[0]
	Y = IV[i].shape[1]
	return sum(sum(IV[i, int(X/3) : int(2*X/3), int(Y/3) : int(2*Y/3)])) * (9/(X * Y))


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


# Number of patches > XY/100 pixels, how many disconnected significantly large regions are present
def f24(i, s):
	count = 0
	for si in s:
		if len(si) >= 164:
			count += 1
	return count


# Number of different color blobs / color complexity of image
def f25(i, dic):
	count = 0
	for key in dic:
		max_length = max(dic[key][0])
		if max_length > 1000:
			count += 1
	return count


# Average Hue value for patch 1
def f26(i, s):
	si = s[0]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 2
def f27(i, s):
	si = s[1]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 3
def f28(i, s):
	si = s[2]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 4
def f29(i, s):
	si = s[3]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Hue value for patch 5
def f30(i, s):
	si = s[4]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IH[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 1
def f31(i, s):
	si = s[0]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 2
def f32(i, s):
	si = s[1]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 3
def f33(i, s):
	si = s[2]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 4
def f34(i, s):
	si = s[3]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Saturation value for patch 5
def f35(i, s):
	si = s[4]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IS[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 1
def f36(i, s):
	si = s[0]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 2
def f37(i, s):
	si = s[1]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 3
def f38(i, s):
	si = s[2]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 4
def f39(i, s):
	si = s[3]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)


# Average Intensity value for patch 5
def f40(i, s):
	si = s[4]
	sum_ = 0
	for pixel in si:
		j, k = pixel
		sum_ += IV[i][j][k]
	return sum_/len(si)

# Measure of largest patch
def f41(i):
	si = s[0]
	return len(si)/16384


def f42(i):
	si = s[1]
	return len(si)/16384

def f43(i):
	si = s[2]
	return len(si)/16384


def f44(i):
	si = s[3]
	return len(si)/16384


def f45(i):
	si = s[4]
	return len(si)/16384


def f46(i, h):
	sumh = 0
	for j in range(5):
		for k in  range(5):
			sumh += abs(h[j] - h[k])
	return sumh

def f47(i, h):
	sumh = 0
	for j in range(5):
		for k in  range(5):
			t = abs(h[j] - h[k])	
			if t < 0.5:
				sumh += 360*t
			else:
				sumh += 360 - 360*t
	return sumh


def f48_pre(i, s):
	centers = []
	for si in s:
		point_sum_x = 0
		point_sum_y = 0
		for point in si:
			x, y = point
			point_sum_x += x
			point_sum_y += y
		x = point_sum_x/len(si)
		y = point_sum_y/len(si)
		centers.append([x,y])
	return centers


def f48(i, s):
	centers = f48_pre(i, s)
	n = 0
	c = centers[n]
	if c[0] < 43:
		r = 10
	elif c[1] < 86:
		r = 20
	else:
		r = 30
	if c[1] < 43:
		cc = 1
	elif c[1] < 86:
		cc = 2
	else:
		cc = 3
	return r + cc


def f49(i, s):
	centers = f48_pre(i, s)
	n = 1
	c = centers[n]
	if c[0] < 43:
		r = 10
	elif c[1] < 86:
		r = 20
	else:
		r = 30
	if c[1] < 43:
		cc = 1
	elif c[1] < 86:
		cc = 2
	else:
		cc = 3
	return r + cc


def f50(i, s):
	centers = f48_pre(i, s)
	n = 2
	c = centers[n]
	if c[0] < 43:
		r = 10
	elif c[1] < 86:
		r = 20
	else:
		r = 30
	if c[1] < 43:
		cc = 1
	elif c[1] < 86:
		cc = 2
	else:
		cc = 3
	return r + cc


def f51(i, s):
	centers = f48_pre(i, s)
	n = 3
	c = centers[n]
	if c[0] < 43:
		r = 10
	elif c[1] < 86:
		r = 20
	else:
		r = 30
	if c[1] < 43:
		cc = 1
	elif c[1] < 86:
		cc = 2
	else:
		cc = 3
	return r + cc


def f52(i, s):
	centers = f48_pre(i, s)
	n = 4
	c = centers[n]
	if c[0] < 43:
		r = 10
	elif c[1] < 86:
		r = 20
	else:
		r = 30
	if c[1] < 43:
		cc = 1
	elif c[1] < 86:
		cc = 2
	else:
		cc = 3
	return r + cc


# DoF feature for Hue property
def f53(i):
	prereq_f10_f11_f12(i)
	v1 = v2 = v3 = 0
	sumv1 = sum(sum(LH[2]))
	if sumv1 > 0:
		v1 = sum(sum(abs(LH[2][4:12,4:12]))) / sumv1
	sumv2 = sum(sum(HL[2]))
	if sumv2 > 0:
		v2 = sum(sum(abs(HL[2][4:12,4:12]))) / sumv2
	sumv3 = sum(sum(HH[2]))
	if sumv3 > 0:
		v3 = sum(sum(abs(HH[2][4:12,4:12]))) / sumv3
	if sumv1 == 0:
		v1 = (v2 + v3)/2
	if sumv2 == 0:
		v2 = (v1 + v3)/2
	if sumv3 == 0:
		v3 = (v1 + v2)/2

	return v1 + v2 + v3


# DoF feature for Saturation property
def f54(i):
	prereq_f13_f14_f15(i)
	v1 = v2 = v3 = 0
	sumv1 = sum(sum(LH[2]))
	if sumv1 > 0:
		v1 = sum(sum(abs(LH[2][4:12,4:12]))) / sumv1
	sumv2 = sum(sum(HL[2]))
	if sumv2 > 0:
		v2 = sum(sum(abs(HL[2][4:12,4:12]))) / sumv2
	sumv3 = sum(sum(HH[2]))
	if sumv3 > 0:
		v3 = sum(sum(abs(HH[2][4:12,4:12]))) / sumv3
	if sumv1 == 0:
		v1 = (v2 + v3)/2
	if sumv2 == 0:
		v2 = (v1 + v3)/2
	if sumv3 == 0:
		v3 = (v1 + v2)/2

	return v1 + v2 + v3


# DoF feature for Intensity property
def f55(i):
	prereq_f16_f17_f18(i)
	v1 = v2 = v3 = 0
	sumv1 = sum(sum(LH[2]))
	if sumv1 > 0:
		v1 = sum(sum(abs(LH[2][4:12,4:12]))) / sumv1
	sumv2 = sum(sum(HL[2]))
	if sumv2 > 0:
		v2 = sum(sum(abs(HL[2][4:12,4:12]))) / sumv2
	sumv3 = sum(sum(HH[2]))
	if sumv3 > 0:
		v3 = sum(sum(abs(HH[2][4:12,4:12]))) / sumv3
	if sumv1 == 0:
		v1 = (v2 + v3)/2
	if sumv2 == 0:
		v2 = (v1 + v3)/2
	if sumv3 == 0:
		v3 = (v1 + v2)/2

	return v1 + v2 + v3


path = "/home/avikalp/semester6/SIGKDD/photo_net_dataset/images/"

if __name__ == '__main__':
	# graph = [[1, 1, 0, 0, 0],
 #            [0, 1, 0, 0, 2],
 #            [1, 0, 0, 2, 2],
 #            [0, 0, 0, 0, 0],
 #            [1, 0, 1, 0, 1]]
	# row = len(graph)
	# col = len(graph[0])
	# g= Graph(row, col, graph)
	# k = 0
	# print ("Number of islands is :",)
	# print(g.countIslands(k))
	# exit()
	subset_indices = list(np.load('good_indices.npy'))
	image_sizes = list(np.load('image_sizes_40p.npy'))
	print('Loading IHSV...')
	IH = np.load('IH_40p.npy')
	IS = np.load('IS_40p.npy')
	IV = np.load('IV_40p.npy')
	print('IV','IHSV loaded.')
	print('Loading LUV...')
	LUV = np.load('LUV_40p.npy')
	print('LUV loaded.')
	feature_vec = []
	for i, index in enumerate(subset_indices):
		print (i)
		feature_vec.append([])
		feature_vec[i].append(f1(i))
		# feature_vec[i].append(f2(i))
		# feature_vec[i].append(f3(i))
		# feature_vec[i].append(f4(i))
		# feature_vec[i].append(f5(i))
		feature_vec[i].append(f6(i))
		# feature_vec[i].append(f7(i))
		# feature_vec[i].append(f8(i))
		# feature_vec[i].append(f9(i))
		# feature_vec[i].append(f10(i))
		# feature_vec[i].append(f11(i))
		# feature_vec[i].append(f12(i))
		# feature_vec[i].append(f13(i))
		# feature_vec[i].append(f14(i))
		feature_vec[i].append(f15(i))
		# feature_vec[i].append(f16(i))
		feature_vec[i].append(f17(i))
		# feature_vec[i].append(f18(i))
		# feature_vec[i].append(f19(i))
		feature_vec[i].append(f20(i))
		feature_vec[i].append(f21(i))
		feature_vec[i].append(f22(i))
		feature_vec[i].append(f23(i))
		
		# print('Starting K-Means')
		# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
		# kmeans.labels_

		# kmeans.predict([[0, 0], [4, 4]])
		_LUV = LUV[i].reshape((16384, 3))
		kmeans = KMeans(n_clusters=kmeans_cluster_num, random_state=0).fit(_LUV)
		# centers = kmeans.cluster_centers_
		graph = kmeans.labels_
		graph = graph.reshape((128,128))
		dic = segmentation(graph)
		s = list(segments(dic))
		H = []
		for k in range(5):
			sumh = 0
			for i1, j1 in s[k]:
				sumh += IH[i][i1][j1]
			H.append(sumh)

		# feature_vec[i].append(f24(i, s))
		feature_vec[i].append(f25(i, dic))
		# feature_vec[i].append(f26(i, s))
		# feature_vec[i].append(f27(i, s))
		feature_vec[i].append(f28(i, s))
		# feature_vec[i].append(f29(i, s))
		# feature_vec[i].append(f30(i, s))
		feature_vec[i].append(f31(i, s))
		# feature_vec[i].append(f32(i, s))
		# feature_vec[i].append(f33(i, s))
		# feature_vec[i].append(f34(i, s))
		# feature_vec[i].append(f35(i, s))
		# feature_vec[i].append(f36(i, s))
		# feature_vec[i].append(f37(i, s))
		# feature_vec[i].append(f38(i, s))
		# feature_vec[i].append(f39(i, s))
		# feature_vec[i].append(f40(i, s))
		# feature_vec[i].append(f41(i))
		# feature_vec[i].append(f42(i))
		feature_vec[i].append(f43(i))
		# feature_vec[i].append(f44(i))
		# feature_vec[i].append(f45(i))
		# feature_vec[i].append(f46(i, H))
		# feature_vec[i].append(f47(i, H))
		# feature_vec[i].append(f48(i, s))
		# feature_vec[i].append(f49(i, s))
		# feature_vec[i].append(f50(i, s))
		# feature_vec[i].append(f51(i, s))
		# feature_vec[i].append(f52(i, s))
		# feature_vec[i].append(f53(i))
		feature_vec[i].append(f54(i))
		# feature_vec[i].append(f55(i))
		# feature_vec[i].append(f56(i))

		# -------------------------- #

		# Do something
		#
		#

		# del feature_vec[i][:]

np.save('selected_feature_vecs.npy', feature_vec)