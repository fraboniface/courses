import numpy as np
import pandas as pd

import utils
from nlp import tfidf

class IGFeaturesGetter():

	def __init__(self, train, labels, df_min=0.01, df_bigram=0.005, tf_bigram=3, igat_unigram=0.01):
		"""Inspiré de http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.2.1938&rep=rep1&type=pdf"""

		self._features = {}

		self._df_min = df_min
		self._df_bigram = df_bigram
		self._tf_bigram = tf_bigram
		self._igat_unigram = igat_unigram

		self._train = train
		self._labels = labels

		self._vocab = utils.get_vocab(train)

		self._n = len(train)
		self._nb0 = list(labels).count(0)
		self._nb1 = list(labels).count(1)

		self._df_unigrams = {}
		self._df_bigrams = {}


	def get_best_bigrams(self):

		self.compute_df()

		B = self.get_bigrams()
		self.compute_df(B=B)

		infogain = self.compute_infogain(False)
		ig_bigram = self.compute_ig_bigram()

		print("Selecting best bigrams...")

		for i,b in enumerate(B):
			for c in [0,1]:
				nb = list(self._labels).count(c)
				removed = False
				if np.sum(self._df_bigrams[b]) < self._df_bigram*nb or np.sum(self._df_bigrams[b]) < self._tf_bigram : # à calculer
					del self._features[b]
					del b
					removed = True

			if not removed and infogain[i] < ig_bigram:
				del self._features[b]
				del b

		return self


	def compute_features(self, corpus=None):

		print("Computing TF-IDF...")

		columns = self._features.keys()

		if corpus is None:

			for index,com in enumerate(self._train):
				for wob in com:
					if wob in columns:
						self._features[wob][index] = tfidf(corpus, wob, com, self._df_unigrams)

				tuples = [(com[i],com[i+1]) for i in range(len(com)-1)]
				for t in tuples:
					if t in columns:
						self._features[t][index] = tfidf(corpus, t, com, self._df_bigrams)

			df = pd.DataFrame(self._features)

		else:

			features, dic = self.compute_df(corpus=corpus)

			for index,com in enumerate(corpus):
				for wob in com:
					if wob in columns:
						features[wob][index] = tfidf(corpus, wob, com, dic)

				tuples = [(com[i],com[i+1]) for i in range(len(com)-1)]
				for t in tuples:
					if t in columns:
						features[t][index] = tfidf(corpus, t, com, dic)
			

			df = pd.DataFrame(features)

		return df.as_matrix()


	def compute_ig_bigram(self):

		ig_unigram = self.compute_infogain()
		sorted_ig = np.sort(ig_unigram)

		ind = int((1-self._igat_unigram)*len(sorted_ig))

		return sorted_ig[ind]


	def get_bigrams(self):

		dic = self._df_unigrams
		B = set()
		for comment in self._train:	
			for i in range(len(comment)-1):
				w1 = comment[i]
				w2 = comment[i+1]
				if np.sum(dic[w1]) > self._df_min or np.sum(dic[w2]) > self._df_min:
					B.update({(w1,w2)})

		return list(B)


	def compute_df(self, corpus=None, B=None):
		"""Marche aussi pour les bigrams a priori.
		Contient [nb avec classe 0, nb avec classe 1]"""

		if B is None:
			if corpus is not None:
				print("Computing unigrams dictionnary...")
				dic = {}
				features = {}
				columns = self._features.keys()
				n = len(corpus)
				for word in columns:
					dic[word] = 0
					for i,com in enumerate(corpus):
						if word in com:
							dic[word] += 1

					features[word] = np.zeros(n)
			else:

				print("Computing unigrams dictionnary...")
				for word in self._vocab:
					self._df_unigrams[word] = [0,0]
					for i,com in enumerate(self._):
						if word in com:
							self._df_unigrams[word][self._labels[i]] += 1

					self._features[word] = np.zeros(self._n)

		else:
			print("Computing bigrams dictionnary...")
			for big in B:
				self._df_bigrams[big] = [0,0]
				for i,com in enumerate(self._train):
					tuples = [(big[i],big[i+1]) for i in range(len(big)-1)]
					if big in tuples:
						self._df_bigrams[big][self._labels[i]] += 1

				self._features[big] = np.zeros(self._n)

		if corpus is None:
			return self
		else:
			return features, dic


	def entropy(self, p):

		return p*np.log(p)


	def compute_infogain(self, unigrams=True):

		print("Computing info gain...")

		if unigrams:
			dic = self._df_unigrams
		else:
			dic = self._df_bigrams

		columns = dic.keys()

		infogain = []
		for i,wob in enumerate(columns):	# wob pour word or bigram
				
			nb0, nb1 = dic[wob]
			nb = nb0+nb1

			# formule à vérifier
			infogain.append(self.entropy(nb0/nb) + self.entropy(nb1/nb) - self.entropy((self._nb0/self._n) - nb0/nb) - self.entropy((self._nb1/self._n) - nb1/nb))

		return np.array(infogain)