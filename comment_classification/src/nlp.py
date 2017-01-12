from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import wordnet as wn

import string
import re

import numpy as np


def tokenize(corpus, auto=True, sentence=False):
	"""Sépare chaque string du tableau en un tableau de ses mots.
	Problème de NLTK : ne sépare pas les mots séparés uniquement par un slash (au moins ça).

	ENTREE
	-----------
	corpus : liste de strings
		Le tableau de commentaires

	auto : booléen
		Si True, le module nltk sera utilisé au lieu de mon implémentation (pourtant admirable).
		La ponctuation apparaîtra alors dans le tableau de commentaires tokenizé.

	sentence : booléen
		Si True, utilse la fonction sent_tokenize de nltk qui sépare en plus les commentaires en phrases.
		2 phrases -> 2 tableaux des phrases tokenizées.

	SORTIE
	-----------
	tokenized_corpus : liste d'listes de strings
		Le tableau des commentaires tokenizé
	"""

	tok_tab = []
	if auto:
		for comment in corpus:
			if sentence:
				tok_tab.append(sent_tokenize(comment))
			else:
				tok = word_tokenize(comment)
				for word in tok:
					if '/' in word: # le tokenizer de nltk ne sépare pas les mots séparé par un slash sans espace
						tok.extend(word.split('/')) # l'ordre des mots n'a pas d'importance donc on rajoute à la fin
						tok.remove(word)
				tok_tab.append(tok)
	else:
		for comment in corpus:

			if comment[0] in string.punctuation:
				tok = re.split('\W+',comment)[1:]
			else:
				tok = re.split('\W+',comment)

			if len(tok) > 1:
				tok_tab.append(tok[:-1])
			else:
				tok_tab.append(tok)

	return tok_tab


def stem(tokenized_corpus):
	"""Réalise le stemming des commentaires en utilisant le SnowballStemmer de NLTK.

	ENTREE
	----------
	tokenized_corpus : array de strings
		Le tableau de commentaires tokenizé.

	SORTIE
	----------
	stemmed_tab : array de strings
		Le tableau de commentaires racinisé.
	"""

	stemmer = SnowballStemmer("english")

	stemmed_corpus = []

	for comment in  tokenized_corpus:
		tab = []
		for word in comment:
			tab.append(stemmer.stem(word))

		stemmed_corpus.append(tab)

	return stemmed_corpus


def lemmatize(tokenized_corpus):
	"""Réalise la lemmatisation des commentaires = stemming plus poussé)"""

	def is_noun(tag):
	    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


	def is_verb(tag):
	    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


	def is_adverb(tag):
	    return tag in ['RB', 'RBR', 'RBS']


	def is_adjective(tag):
	    return tag in ['JJ', 'JJR', 'JJS']


	def penn_to_wn(tag):
	    if is_adjective(tag):
	        return wn.ADJ
	    elif is_noun(tag):
	        return wn.NOUN
	    elif is_adverb(tag):
	        return wn.ADV
	    elif is_verb(tag):
	        return wn.VERB
	    return None


	lemmatizer = WordNetLemmatizer()

	lemmatized_corpus = []
	for comment in tokenized_corpus:
		tagged = pos_tag(comment)
		com = []
		for t in tagged:
			if penn_to_wn(t[1]) != None:
				com.append(lemmatizer.lemmatize(t[0], penn_to_wn(t[1])))
			else:
				com.append(t[0])

		lemmatized_corpus.append(com)

	return lemmatized_corpus



def tfidf(tokenized_corpus, word, comment, df_dict):
	"""Retourne la TF-IDF de word dans comment."""

	n = len(tokenized_corpus)

	if type(word).__name__ == 'tuple':
		tuples = [(comment[i],comment[i+1]) for i in range(len(comment)-1)]
		tf = tuples.count(word)
	else:
		tf = comment.count(word)

	df = df_dict[word]
	if type(df).__name__ == 'list':
		df = np.sum(df)

	tfidf = tf*np.log2(n/df)

	return tfidf