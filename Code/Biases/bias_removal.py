import csv
import numpy as np
# good_indices = np.load('subset60p_indices.npy')
feature_vecs = np.load('normalised_feature_vecs.npy')
scores = np.load('scores_before_bremoval.npy')
# print(scores)
a = np.load('unbias.npy')
a = list(a)

anims = []
for i in a:
	anims.append(scores[i-1])
print (np.mean(anims))
print (np.std(anims))
# print(a)
