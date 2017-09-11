import csv
import os
import math
import numpy as np

# reader = csv.reader(open('/home/avikalp/semester6/SIGKDD/implementation/image_data/output5_norm_score.csv', 'r'))

ocr_list = ["discount ", "offer ", "scheme", "retweet", "rt", "off ", "sale", "% "]
tweet_list = ["#national", "#international"]
neural_list = ["cat ", "dog ", "bird "]
neural_human = ["human ", "man ", "woman "]
dates = ["2016-12-19", "2016-12-20", "2016-12-21", "2016-12-22", "2016-12-23", "2016-12-24", "2016-12-25", "2016-12-26", 
"2016-12-27", "2016-12-28", "2016-12-29", "2016-12-30", "2016-12-31", "2017-12-1", "2016-10-25", "2016-10-26"
"2016-10-27", "2016-10-28", "2016-10-29", "2016-10-30", "2016-10-31", "2016-11-28", "2016-11-25"]
bias_list = []

# human = []
# animal = []
# sale = []
# tweet = []
# holiday = []



# tweet_count = 0
# ocr_count = 0
# neural_count = 0
# neural_human_count = 0
# count = 1
# date_count = 0
# for row in reader:
# 	date = row[3]
# 	tweet_text = row[8].lower()
# 	ocr_text = row[9].lower()
# 	neural_text = row[11].lower()
# 	if any(x in date for x in dates):
# 		holiday.append(count)
# 		date_count += 1
# 	elif any(x in neural_text for x in neural_human):
# 		human.append(count)
# 		neural_human_count += 1
# 	elif any(x in tweet_text for x in tweet_list):
# 		tweet.append(count)
# 		tweet_count += 1
# 	elif any(x in ocr_text for x in ocr_list):
# 		sale.append(count)
# 		ocr_count += 1
# 	elif any(x in neural_text for x in neural_list):
# 		animal.append(count)
# 		neural_count += 1
# 	count += 1
# print("Bias split : \nOCR : " + str(ocr_count) + "\nTweet Text : " + str(tweet_count) + "\nSpecial Occassions " + str(date_count))
# print("Human vs Animal split : " + "\nHuman : " + str(neural_human_count) + "\nAnimal : " + str(neural_count))
# # print("The length of the list is " + str(len(bias_list)))
# # print(len(bias_list))
# np.save('human_bias.npy', human)
# np.save('animal_bias.npy', animal)
# np.save('tweet_bias.npy', tweet)
# np.save('sale_bias.npy', sale)
# np.save('holiday_bias.npy', holiday)

# for i in bias_list:
# 	print(i)

# h = list(np.load('human_bias.npy'))
# a = list(np.load('animal_bias.npy'))
# t = list(np.load('tweet_bias.npy'))
# s = list(np.load('sale_bias.npy'))
# hol = list(np.load('holiday_bias.npy'))

# bias = h +  a + t + s + hol
# unbias = [x for x in range(7963)]
# unbias.remove(0)
# for i in bias:
# 	unbias.remove(i)
# np.save('unbias.npy', unbias)
# np.save('bias.npy', bias)



reader = csv.reader(open('/home/avikalp/semester6/SIGKDD/implementation/image_data/output5_norm_score.csv', 'r'))
bias = np.load('unbias.npy')
unbias = np.load('bias.npy')

print(bias)
print(unbias)

c = 1
bias_score = []
bias_count = 0

unbias_score = []
unbias_count = 0

for row in reader:
	if c in bias:
		bias_score.append(float(row[12]))
		bias_count += 1
	elif c in unbias:
		unbias_score.append(float(row[12]))
		unbias_count += 1
	c += 1

print(len(bias_score))



np.save('bias_scores.npy', bias_score)
np.save('unbias_scores.npy', unbias_score)
print(np.mean(bias_score), np.std(bias_score))
print(np.mean(unbias_score), np.std(unbias_score))





