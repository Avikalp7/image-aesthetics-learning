import tweepy
import wget
from tweepy import OAuthHandler
import json
import shutil
import csv
import os
consumer_key = '<We can\'t make our key public here, please enter your own key here :D :)'
consumer_secret = '...'
access_token = '...'
access_secret = '...'


from pandas import DataFrame
# from tweet import compute
from math import exp


@classmethod
def parse(cls, api, raw):
	status = cls.first_parse(api, raw)
	setattr(status, 'json', json.dumps(raw))
	return status


metadata = open("output2.csv", 'a')
wr = csv.writer(metadata,dialect='excel')
# Status() is the data model for a tweet
tweepy.models.Status.first_parse = tweepy.models.Status.parse
# tweepy.models.Status.parse = parse
# User() is the data model for a user profil
tweepy.models.User.first_parse = tweepy.models.User.parse
# tweepy.models.User.parse = parse
# You need to do it for all the models you need

def compute(name, count_num, max_num):
	tweepy.models.Status.parse = parse
	tweepy.models.User.parse = parse
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_secret)
	print("Authentication successful")
	api = tweepy.API(auth)
	tweets = api.user_timeline(screen_name= name,
							   count=200, include_rts=False,
							   exclude_replies=True)
	# print(tweets)
	if len(tweets) == 0:
		last_id = 0
	else:
		last_id = tweets[-1].id
	print("Successfully downloaded the tweet information")
	count = 0
	while (1):
		count = count + 1
		try:
			more_tweets = api.user_timeline(screen_name=name,
											count=200,
											include_rts=False,
											exclude_replies=True,
											max_id=last_id - 1)
		except:
			print ('Exception Handled!')
			continue
		# There are no more tweets
		if (len(more_tweets) == 0):
			break
		else:
			last_id = more_tweets[-1].id - 1
			tweets = tweets + more_tweets
	media_files = []
	for status in tweets:
		if len(media_files) >= max_num:
			break
		if 'media' in status.entities:
			media = status.entities['media']
			if (media[0]['type'] == 'photo' and 'photo' in status.entities['media'][0]['expanded_url']):
				list = [str(status.author.name), str(status.retweet_count), str(status.favorite_count), str(status.created_at),
							   str(status.user.followers_count),str(status.user.friends_count), (status.user.location).encode('utf-8'),
							   (status.entities['media'][0]['expanded_url']).encode('utf-8'), (status.text).encode('utf-8')]
				wr.writerow(list)
				print(list)
				print('\n\n')
				print(media[0]['media_url'])
				media_files.append(media[0]['media_url'])
			else:
				continue	
	# print media_files
	count = count_num
	print("\nNumber of images downloaded is " + str(len(media_files)))
	print("\nImage url extraction successful")
	folder_name = 'twitter_images'
	if(not os.path.isdir(os.path.join(os.getcwd()))):
		os.makedirs(folder_name)
	for media_file in media_files:
		count = count + 1
		if(count%5 == 0):
			print("\n" + str(count) + " images have been downloaded")
		try:
			filename = wget.download(media_file)
		except:
			print("\nException Handled!")
			continue
		shutil.copyfile(filename, "." + "/" + folder_name + "/img" + str(count))
		os.remove(filename)
	return count



def sigmoid(x):
	return 1/float(1+exp(-x))

def preprocess():
	df = DataFrame.from_csv('names.csv', header = 0)
	num = []
	# print (df.keys)
	temp = list(df['Tweets(K)'])
	temp2 = list(df['Organisation_Handle'])
	# print (temp[0])
	# temp.remove('Tweets(K)')
	mean = sum(temp)/float(len(temp))
	count = 0
	max_elem = max(temp)
	min_elem = min(temp)
	for x in temp:
		temp[count] = (temp[count] - mean)/float(max_elem - min_elem)
		count = count + 1
	for i in range(0, len(temp)):
		j = int(300*sigmoid(temp[i]))
		num.append(j)
	return num, temp2




# names = ['Audi', 'BMW', 'CocaCola', 'drpepper', 'subway']
num_imgs, names = preprocess()
current_count = 0
count = 0
for name in names[0:18]:
	print('Starting with following corp: ')
	print(name)
	current_count = compute(name, current_count, num_imgs[count])
	count = count + 1
