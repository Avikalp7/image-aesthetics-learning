import numpy as np

t = np.load('normalised_feature_vecs.npy')
print('LEN: ',len(t[0]))
good_indices = np.load('subset60p_indices.npy')
i = 31
maxi = -1
maxind = 0
for n,vec in enumerate(t):
	if n==2967:
		continue
	if vec[i] > maxi:
		maxind = n
		maxi = vec[i]
print(maxind)
print(t[maxind][i])
print(good_indices[maxind])

