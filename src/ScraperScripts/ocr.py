"""
Extracting text from images with Tesseract OCR
To run this code first install the pytesseract library from git
"""
import pytesseract
from PIL import Image

import os
import csv
import datetime



def main():
	"""
	Takes user input for image folder and uses pytesseract to get text, the input data csv is input.csv,
	a new csv with an appended column contating ocr text will be generated and named output.csv
	"""
	print("Note that your current csv should be named input.csv and you will get output.csv with the appended column")
	ispresent = {}
	list_of_required = np.load("../../data/dicount_wordlist.npy")
	folder_name = raw_input("Enter the name of the folder containing images in the current directory: \n")
	direc_path = str(os.getcwd() + "/" + str(folder_name))
	dir = os.getcwd() + "/" + str(folder_name); 
	onlyfiles = next(os.walk(dir))[2]
	image_count = 0
	maximum = -1
	for com in onlyfiles:
		if maximum < int(com[3:]):
			maximum = int(com[3:])
		ispresent[int(com[3:])] = 1
	for i in range(1, maximum + 1):
		if i in ispresent:
			image_count = image_count + 1
			ispresent[i] = 1
		else:
			ispresent[i] = 0
	print "Total number of images are  " + str(image_count)
	reader = csv.reader(open('input.csv', 'r'))
	writer = csv.writer(open('output.csv','w'))
	count = 1
	sale_count = 0
	for row in reader:
		if(ispresent[count]	== 1):
			parsed = str(pytesseract.image_to_string(Image.open(str(direc_path + "/" + "img" + str(count))), lang='eng'))
			row.append(parsed)
			if any(x in parsed for x in list_of_required):
				sale_count = sale_count + 1
			if count % 5 == 0 :
				print "sale frequency " + str(sale_count) + " out of " + str(count) + " images scanned"
		else:
			row.append('NO_IMAGE')
		writer.writerow(row)
		count = count + 1
	return


if __name__ == "__main__":
	main()
