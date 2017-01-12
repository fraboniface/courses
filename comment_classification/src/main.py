# coding: utf8

import numpy as np
import pandas as pd

import utils
import nlp

from logreg import LogisticRegression
from cross_val import cv_score
#from NaiveBayesClassifier import NBC
#from NBFeaturesGetter import NBFeaturesGetter
from IGFeaturesGetter import IGFeaturesGetter

def main():

	X, y = utils.load_comments('train.csv')
	X_test = utils.load_comments('test.csv', test=True)

	#X, punct_train, maj_train, bw_train = utils.process(X)
	X = utils.process(X)
	#X_test, punct_test, maj_test, bw_test = utils.process(X_test)
	X_test = utils.process(X_test)

	#X_train, y_train = X[::2], y[::2]
	#X_val, y_val = X[1::2], y[1::2]
	fg = IGFeaturesGetter(X, y)
	#fg.compute_columns()
	X = fg.compute_features()
	X_test = fg.compute_features(X_test)
	#clf = NBC(vocab=voc)
	#clf.fit(X_train,y_train)
	#print(clf.score(X_val, y_val))

	#df = utils.get_features(X)
	#df_train, df_test = utils.get_train_test_features(X, X_test)
	#df_test = utils.get_features(X_test, auto=False)
	#df['nb_punct'] = punct_train
	#df['nb_maj'] = maj_train
	#df['nb_bw'] = bw_train

	#X = df.as_matrix()
	#X_train = df_train.as_matrix()
	#X_test = df_test.as_matrix()
	#print("X :", X.shape)
	#print(X_train.shape)
	#print(X_test.shape)

	clf = LogisticRegression()
	#scores = cv_score(clf, X, y, cv=5)
	#print(scores)
	#print(scores.mean())
	clf.fit(X,y)
	#print(clf.score(X_val, y_val))
	#print(clf.coeff)
	#print(clf.intercept)
	#print(clf.score(X,y))
	pred = clf.predict(X_test)

	np.savetxt('pred.txt', pred, fmt='%s')

if __name__ == '__main__':
	main()