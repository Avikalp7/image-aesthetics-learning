from __future__ import division
import csv
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from scipy.stats import norm
from scipy import stats
FLAGS = re.MULTILINE | re.DOTALL

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def hashtag(text):
	text = text.group()
	hashtag_body = text[1:]
	print(hashtag_body)
	if hashtag_body.isupper():
		result = "<hashtag> {} <allcaps>".format(hashtag_body)
	else:
		# result = " ".join(["<hashtag>"] + re.split(r"([A-Z_]?)", hashtag_body, flags=FLAGS))
		result = " ".join(["<hashtag>"] + camel_case_split(hashtag_body))
		# print(result)
	return result



def bias(text):
	pretext = text
	text = re.sub(r"#\S+", hashtag, text, flags=FLAGS)
	if pretext != text:
		if ('<hashtag> National' in text) or ('<hashtag> International') in pretext:
			print('hashtag found') 
			return True
	sale_words = ['off', '%', 'sale', 'discount', 'offer', 'retweet', 'win', 'chance', 'gain', 'cat', 'dog']
	for w in sale_words:
		if w in text:
			return True


def reject_outliers(x, data, m=6):
	# return data[abs(data - np.mean(data)) < m * np.std(data)]
	# indices = abs(data - np.mean(data)) < m * np.std(data)
	# print (indices)
	return x[abs(data - np.mean(data)) < m * np.std(data)], data[abs(data - np.mean(data)) < m * np.std(data)]
    # data[abs(data - np.mean(data)) < m * np.std(data)]


# ALL IMAGES #
reader = csv.reader(open('output5_norm_score.csv', 'r'))
# alll = []

# dic = {}
# j = 0
# for row in reader:
# 	e = int(row[1]) + int(row[2])
# 	nf = int(row[4])
# 	try:
# 		dic[nf]
# 		dic[nf].append(e)
# 	except:
# 		dic[nf] = [e]
# with open('dic.pickle', 'wb') as handle:
#     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('dic.pickle', 'rb') as handle:
#     b = pickle.load(handle)

# ALL IMAGES END #



# reader = csv.reader(open('output5_norm_score.csv', 'r'))
# scores = []
# for row in reader:
# 	scores.append(float(row[12]))
# np.save('scores_before_bremoval.npy', scores)









# SCORE NORM #
# dic = {}
# for row in reader:
# 	try:
# 		dic[row[0]]
# 		dic[row[0]].append(int(row[1]) + int(row[2]))
# 	except:
# 		dic[row[0]] = [int(row[1]) + int(row[2])]

# dic = {}
# for row in reader:
# 	try:
# 		dic[row[0]]
# 		dic[row[0]].append(float(row[12]))
# 	except:
# 		try:
# 			dic[row[0]] = [float(row[12])]
# 		except:
# 			print(row)

# mean = {}
# std = {}
# for key in dic:
# 	mean[key] = np.mean(dic[key])
# 	std[key] = np.std(dic[key])
# 	print(mean[key], std[key])

# alll = []
# reader = csv.reader(open('output5.csv', 'r'))
# for row in reader:
# 	score = int(row[1]) + int(row[2])
# 	# print(mean[row[0]], std[row[0]])
# 	if(std[row[0]]>0):
# 		score = (score - mean[row[0]])/std[row[0]]
# 	else:
# 		score = (score - mean[row[0]])/(std[row[0]]+0.01)
# 	try:
# 		float(score)
# 	except:
# 		print(score, row)
# 	row.append(score)
# 	alll.append(row)

# writer = csv.writer(open('output5_norm_score.csv','w'))
# writer.writerows(alll)



# SCORE NORM #




# Animal Bias #
# reader = csv.reader(open('output5_norm_score.csv', 'r'))
# alll = []

# dic = {}
# lis = []
# j = 0
# for row in reader:
# 	e = float(row[12]) 
# 	text = row[11]
# 	if "dog " in text or "cat " in text or "bird " in text or "horse " in text:
# 		lis.append(e)

# print ('LENGTH: ',len(lis))
# print (lis)
# mu, std = norm.fit(lis)
# W, PW = stats.shapiro(x=np.array(lis))
# print('SHAPIRO: ', W, PW)


# plt.hist(lis, bins=30, normed=True, alpha=0.6, color='g', range=(-5,5))
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)



# plt.show()

# Animal Bias End #



# reader = csv.reader(open('output5_norm_score.csv', 'r'))
# alll = []

# dic = {}
# lis = []
# j = 0
# for row in reader:
# 	if row[0] == 'PUMA':
# 		e = float(row[12]) 
# 		# text = row[11]
# 		lis.append(e)	
	
# 	# if "dog " in text or "cat " in text or "bird " in text or "horse " in text:
# 	# 	lis.append(e)

# print ('LENGTH: ',len(lis))
# print (lis)
# mu, std = norm.fit(lis)
# W, PW = stats.shapiro(x=np.array(lis))
# print('SHAPIRO: ', W, PW)


# plt.hist(lis, bins=30, normed=True, alpha=0.6, color='g', range=(-5,5))
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.show()



dic = {}
for row in reader:
	try:
		dic[row[4]]
		dic[row[4]].append(int(row[1])+int(row[2]))
	except:
		dic[row[4]] = [int(row[1])+int(row[2])]

mean = {}
for key in dic:
	mean[key] = np.mean(dic[key])




means = []
nfs = []
for key in dic:
	# mean[key] = sum(dic[key])/len(dic[key])
	# if key > 3000000:
	# 	continue
	if len(dic[key]) < 30:
		continue
	if bias(row[8]):
		continue
	mean = sum(dic[key])/len(dic[key])
	# if key > 2500000:
	# 	print (key)
	# 	print (mean)
	# for element in dic[key]:
	# 	if element > mean + 0.2*mean or element < mean - 0.2*mean:
	# 		dic[key].remove(element)

	mean = sum(dic[key])/len(dic[key])
	# if mean > 3000:
	# 	print ('Key: ', key)
	# 	continue
	means.append(mean)
	nfs.append(int(key))

# nfs, means = reject_outliers(np.array(nfs), np.array(means))
nfs = list(nfs)
means = list(means)

print(nfs)
print(means)

from scipy.stats.stats import pearsonr
print (pearsonr(nfs,means))
print (stats.spearmanr(nfs, means))

# # lists = sorted(dic.items()) # sorted by key, return a list of tuples

# # means, nfs = zip(*lists) # unpack a list of pairs into two tuples
# for i in range(len())


list1, list2 = zip(*sorted(zip(nfs, means)))


plt.plot(list1, list2, 'bo')
plt.xlim(0,7000000)
plt.xlabel('Number of Page F')
plt.ylim(0,5000)
plt.show()













# # mean = {}
# means = []
# nfs = []
# for key in dic:
# 	# mean[key] = sum(dic[key])/len(dic[key])
# 	if key > 3000000:
# 		continue
# 	if len(dic[key]) < 20:
# 		continue
# 	if bias(row[8]):
# 		continue
# 	mean = sum(dic[key])/len(dic[key])
# 	if key > 2500000:
# 		print (key)
# 		print (mean)
# 	# for element in dic[key]:
# 	# 	if element > mean + 0.2*mean or element < mean - 0.2*mean:
# 	# 		dic[key].remove(element)

# 	mean = sum(dic[key])/len(dic[key])
# 	if mean > 3000:
# 		print ('Key: ', key)
# 		continue
# 	means.append(mean)
# 	nfs.append(key)

# nfs, means = reject_outliers(np.array(nfs), np.array(means))

# # lists = sorted(dic.items()) # sorted by key, return a list of tuples

# # means, nfs = zip(*lists) # unpack a list of pairs into two tuples




# plt.plot(nfs, means, 'ro')
# plt.show()




















# lists = sorted(d.items()) # sorted by key, return a list of tuples

# x, y = zip(*lists) # unpack a list of pairs into two tuples

# plt.plot(x, y)
# plt.show()