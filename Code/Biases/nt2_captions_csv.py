import csv
import os.path
import numpy as np

import pickle

with open('captions.pkl', 'rb') as handle:
    dic = pickle.load(handle)

reader = csv.reader(open('output4.csv', 'r'))
writer = csv.writer(open('output5.csv','w'))

row = next(reader)
alll = []

j = 1
for row in reader:
	try:
		t = dic[str(j)]
		row.append(t)
	except KeyError:
		pass
	alll.append(row)
	j += 1

writer.writerows(alll)
