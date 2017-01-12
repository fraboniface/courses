import numpy as np
import pandas as pd

import utils
from nlp import tfidf

class NBFeaturesGetter():

	def __init__(self, train, labels, nb_chosen=100, tfidf=True):
		"""Sélectionne les meilleurs bigrammes en se basant sur la log-likelihood des unigrammes."""

		self._nb_chosen = nb_chosen
		self._tfidf = tfidf

		self._train = train
		self._labels = labels


	def compute_columns(self):

		print("Computing columns...")

		vocab = utils.get_vocab(self._train)
		log_lh = self.compute_log_likelihood(vocab)

		indices = np.argsort(log_lh)
		sorted_lh = np.sort(log_lh)
		sorted_voc = vocab[indices]

		bigrams = self.get_best_bigrams(sorted_voc[-self._nb_chosen:]) # on ne passe que les meilleurs mots

		self._columns = set(vocab)
		self._columns.update(bigrams)
		self._columns = list(self._columns)

		return self


	def compute_features(self, corpus=None):

		if corpus is None:
			corpus = self._train

		n = len(corpus)

		df_dict = {}
		features = {}

		print("Computing dictionnary...")
		for wob in self._columns:
			if type(wob).__name__ == 'tuple':
				cnt = 0
				for com in corpus:
					tuples = [(wob[i],wob[i+1]) for i in range(len(wob)-1)]
					if wob in tuples:
						cnt += 1

				df_dict[wob] = cnt
			else:
				tmp = len([com for com in corpus if wob in com])
				"""if tmp > 2:
					df_dict[word] = tmp
					features[word] = np.zeros(n)"""
				df_dict[wob] = tmp

			features[wob] = np.zeros(n)

		print("Computing TF-IDF...")
		for index,com in enumerate(corpus):
			for word in com:
				if word in self._columns:
					features[word][index] = tfidf(corpus, word, com, df_dict)

			tuples = [(wob[i],wob[i+1]) for i in range(len(wob)-1)]
			for t in tuples:
				if t in self._columns:
					features[t][index] = tfidf(corpus, t, com, df_dict)

		df = pd.DataFrame(features)

		return df.as_matrix()
		

	def compute_log_likelihood(self, vocab):

		log_lh = []
		for word in vocab:
			cnt0 = 0
			cnt1 = 0
			for i,com in enumerate(self._train):
				if type(word).__name__ == 'list':
					tuples = [(com[i],com[i+1]) for i in range(len(com)-1)]
					if word in tuples:
						if self._labels[i] == 0:
							cnt0 += 1
						else:
							cnt1 += 1
				else:
					if word in com:
						if self._labels[i] == 0:
							cnt0 += 1
						else:
							cnt1 += 1

			log_lh.append(np.log(cnt1/(cnt0+cnt1)))

		return np.array(log_lh)


	def get_best_bigrams(self, words):

		B = []
		for comment in self._train:
			l = len(comment)
			if l > 1:
				for i in range(l):
					w = comment[i]
					if w in words:
						if i == 0:
							B.append((w, comment[1]))
						elif i == l-1:
							B.append((comment[l-2],w))
						else:
							w1 = comment[i-1]
							w2 = comment[i+1]
							B.append([(w1,w), (w,w2)][0])

		return B