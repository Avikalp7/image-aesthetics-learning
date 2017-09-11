import csv
import os.path
import numpy as np

# import pickle

# with open('captions.pkl', 'rb') as handle:
    # dic = pickle.load(handle)

reader = csv.reader(open('output5_Edit.csv', 'r'))
# writer = csv.writer(open('output5.csv','w'))

# row = next(reader)
alll = []

j = 0
for row in reader:
	if row[12] == '-1':
		j+=1
print(j)	

# writer.writerows(alll)
