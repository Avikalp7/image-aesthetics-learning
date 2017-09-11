import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import spline
import scipy.special as sps

def func(x_values, maximum_y, maximum_x, sigma): 
    t = maximum_y*np.exp(-(x_values-np.mean(x_values))**2/(2*sigma**2))
    # print('t: ',t)
    return t


def reject_outliers(data, m=2):
	# return data[abs(data - np.mean(data)) < m * np.std(data)]
	# indices = abs(data - np.mean(data)) < m * np.std(data)
	# print (indices)
	return data[abs(data - np.mean(data)) < m * np.std(data)]

def plot(lis):
	data = lis
	# Fit a normal distribution to the data:
	print(data)
	# mu, std = norm.fit(data)
	# sigma = std

	# data = [x for x in data if -1.5 <  x < 1.2 ]
	mu = np.mean(lis)
	sigma = np.std(lis)
	print(mu,sigma)
	# sigma = 0.01
	# sigma = 0.01

	# Plot the histogram.
	# print(data)

	n, bins, patches = plt.hist(data, bins=40,color='blue',normed=True,range = (-1.5, 1.5))
	bin_centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
	# print(len(bin_centers), len(n))
	# exit()
	x = bin_centers
	y = n

	xnew = np.linspace(x.min(),x.max(),100) #300 represents number of points to make between T.min and T.max

	y_smooth = spline(x,y,xnew)


	#------------
	# pi = np.poly1d(np.polyfit(xnew, y_smooth, 4))
	
	# # plt.clf()
	
	# _ = plt.plot(x, y, '.', x, pi(x), 'k', linewidth=2.5)#, xp, p30(xp), '--')
	#-------------


	#NORMAL
	# plt.clf()
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
	
	#GAMMA
	# shape = 2
	# scale = 2
	# y = bins**(shape-1)*(np.exp(-bins/scale) /
 #                     (sps.gamma(shape)*scale**shape))
	# plt.plot(bins, y, linewidth=2, color='r')

	# plt.ylim(-2,2)
	# plt.xlim(-2,2)


	# Plot the PDF.
	# xmin, xmax = plt.xlim()
	# print(xmin, xmax)
	# t = np.random.normal(mu, std, 300)
	# x = np.linspace(-1.5, 1.5, 100)
	# p = norm.pdf(x, mu, std)
	# plt.plot(x, p, 'k', linewidth=2)
	# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
	# plt.title(title)
	# popt, pcov = curve_fit(func, bin_centers, n)
	# #plt.hist(myData) # Not necessary if you haven't showed it before.
	# print('popt: ', popt)
	# plt.line(bin_centers, func(bin_centers, *popt))
# 	plt.show()
# plot([1,2,3,4,4,5,5,5,6,6,7,8,9,10])
a = np.load('human_bias.npy')
s = np.load('scores_before_bremoval.npy')
lisa = []
for index in a:
	lisa.append(s[index-1])
# print(a)
lisa = reject_outliers(np.array(lisa))

# lisa = np.random.gamma(2, 2, 1000)
# print('\n\nLength: ', len(lisa), ' Mean: ', np.mean(s))
plot(lisa)
plt.show()
print(np.mean(lisa))