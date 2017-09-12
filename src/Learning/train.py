from sklearn import svm
import numpy as np
import pickle


def classifier_data(y, lower_threshold = 0.4, upper_threshold = 0.6):
	for i, score in enumerate(y):
		if y <= lower_threshold:
			y[i] = 0
		elif y >= upper_threshold:
			y[i] = 1


def load_data(classify_data = False):
	X = np.load('../../Data/normalised_feature_vecs.npy')
	m = X.shape[0]
	n = X.shape[1]
	y = np.load('scores.npy')
	if classify_data:
		classifier_data(y)

	training_indices = np.random.choice(m, int(0.8*m), replace = False)
	trX = X[training_indices]
	trY = y[training_indices]
	
	test_indices = []
	for index in range(m):
		if index not in training_indices:
			test_indices.append(index)
	tsX = X[test_indices]
	tsY = y[test_indices]

	return trX, trY, tsX, tsY 


def save_model(clf):
	pickle.dump(clf, open( "model.p", "wb" ) )


if __name__ == '__main__':
	trainX, trainY, testX, testY = load_data()
	classify_data = False

	if classify_data:
		clf = svm.SVC(C = 1, gamma = 3.7)
	else:
		clf = svm.SVR(C = 1, gamma = 3.7)
	
	clf.fit(trainX, trainY)
	
	# Saving model
	save_model(clf)

	# Reading model
	clf = pickle.load( open( "../../Data/model.p", "rb" ) )

	prediction = clf.predict(testX)
	
	accuracy = np.mean((prediction == testY)) * 100.0
    print ("\nTest accuracy: %lf%%" % accuracy) 