import numpy as np

import urllib.request
import urllib
import csv



def main():
	# Reading the data csv with urls for images in photo-net dataset
	reader = csv.reader(open('../../Data/photonet_dataset/photonet_data.csv', 'r'))
	row = next(reader)
	
	url_base = "http://gallery.photo.net/photo/"
	url_end = "-md.jpg"
	path = "../../Data/photonet_dataset/images/"
	
	# Indices for which image fails to download
	bad_indices = []

	j = 0
	for row in reader:
		j += 1
		url = url_base + row[1] + url_end
		filename = (path + str(j) + ".jpg") 

		bad_count = 0
		while True:
			if bad_count >= 4:
				print('Bad Index: ', j)
				bad_indices.append(j)
				break
			try:
				request = urllib.request.urlopen(url, timeout=10)
			except Exception as e:
				bad_count += 1
				print("Exception")
				print(e)
				continue

			with open(filename, 'wb') as f:
			    try:
			        f.write(request.read())
			        break
			    except Exception as e:
			    	bad_count += 1
			    	print("Error")
		print(j)
		if j >= 3500:
			break

	# Save the indices for which image download failed
	np.save('bad_indices.npy', bad_indices)
	print("Number of images for which download failed: ", len(bad_indices))
	return


if __name__ == "__main__":
	main()