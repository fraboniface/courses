import numpy as np
import pandas as pd

import string
import re

from nlp import tokenize, tfidf, stem, lemmatize

def load_comments(filename, test=False):
	"""Récupère les commentaires ainsi que leur label si train=True (défaut).

	ENTREE
	-----------
	filename : string
		Le nom du fichier

	test : booleen
		Si test vaut True, la fonction renvoit la variable cible. Sinon elle renvoit seulement les commentaires.

	SORTIE
	----------
	X : liste de strings
		Le tableau de commentaire

	y : liste de booleens
		La variable cible renvoyée si test=False.
	"""

	print("Loading comments")

	X=[]
	with open(filename, "r") as f:
		if not test:
			y=[]
			for line in f:
				y.append(int(line[0]))
				X.append(line[5:-4])

			y = np.array(y)
			return X, y
		else:
			for line in f:
				X.append(line[1:-1])
				
			return X


def process(corpus, auto=True, dolem=False, dostem=True):
	"""Applique clean, tokenize, remove_stop_words_punctuation et stem au corpus"""

	corp = clean(corpus)
	print("Cleaning done")

	corp = tokenize(corp, auto)
	print("Tokenizing done")

	#punct_stats = count_punctuation(corp, ratio=True, auto=auto)
	#maj_stats = count_capitals(corp, auto)
	#bw_stats = get_bw_stats(corp, ratio=True)

	corp = remove_stop_words_punctuation(corp)
	print("Removing done")

	if dolem:
		corp = lemmatize(corp)
		print('Lemmatizing done')
	elif dostem:
		corp = stem(corp)
		print("Stemming done")

	#return corp, punct_stats, maj_stats, bw_stats
	return corp


def get_features(corpus):
	"""Retourne un dataframe des features.

	ENTREE
	----------
	corpus : liste de strings
		Le tableau des commentaires passé par la fonction process

	SORTIE
	----------
	df : pandas DataFrame
		DataFrame contenant les mots du vocabulaire en colonne et les commentaires en ligne.
	"""
	"""
	n = len(tokenized_corpus)
	lengths = []

	keys = ['_url', '_nb', '_date', '_rep', '_happysmiley', '_sadsmiley', '_angrysmiley']

	features = {}
	for key in keys:
		features[key] = np.zeros(n)

	for i,com in enumerate(tokenized_corpus):
		length = 0
		for word in com:
			length += len(word) # ne marche que si l'on a pas tokenizé par phrase
			if word in keys:
				features[word][i] += 1

		lengths.append(length)

	lengths = np.array(lengths)

	df = pd.DataFrame(features)

	new_keys = ['nb_url', 'nb_nb', 'nb_date', 'nb_rep', 'nb_hsm', 'nb_ssm', 'nb_asm']
	df.columns = new_keys

	_, maj_stats, pun_stats, bw_stats = get_statistics(tokenized_corpus)

	df['length'] = lengths
	df['maj_ratio'] = maj_stats
	df['punct_ratio'] = pun_stats
	df['bw_ratio'] = bw_stats
	"""

	#corp, punct_stats, maj_stats, bw_stats = process(corpus, auto)
	#corp = process(corpus, auto)

	n = len(corpus)

	vocab = get_vocab(corpus)

	df_dict = {}
	features = {}
	print("Computing dictionnary...")
	for word in vocab:
		tmp = len([com for com in corpus if word in com])
		"""if tmp > 2:
			df_dict[word] = tmp
			features[word] = np.zeros(n)"""
		df_dict[word] = tmp
		features[word] = np.zeros(n)

	print("Computing TF-IDF...")
	for index,com in enumerate(corpus):
		for word in com:
			if word in features.keys():
				features[word][index] = tfidf(corpus, word, com, df_dict)

	df = pd.DataFrame(features)

	return df.as_matrix()


def get_train_test_features(train, test):
	"""Renvoit les features à la fois pour le train et pour le test."""

	n_train = len(train)
	n_test = len(test)

	vocab = get_cross_vocab(train, test)

	df_dict_train = {}
	df_dict_test = {}

	train_features = {}
	test_features = {}
	print("Computing dictionnaries...")
	for word in vocab:
		tmp_train = len([com for com in train if word in com])
		tmp_test = len([com for com in test if word in com])

		"""if tmp > 2:
			df_dict[word] = tmp
			features[word] = np.zeros(n)"""
		df_dict_train[word] = tmp_train
		df_dict_test[word] = tmp_test
		train_features[word] = np.zeros(n_train)
		test_features[word] = np.zeros(n_test)

	print("Computing TF-IDF in train...")
	for index,com in enumerate(train):
		for word in com:
			if word in train_features.keys():
				train_features[word][index] = tfidf(train, word, com, df_dict_train)

	print("Computing TF-IDF in test...")
	for index,com in enumerate(test):
		for word in com:
			if word in test_features.keys():
				test_features[word][index] = tfidf(test, word, com, df_dict_test)

	df_train  = pd.DataFrame(train_features)
	df_test = pd.DataFrame(test_features)

	return df_train, df_test


def load_stop_words():
	"""Retourne le tableau des stop words en anglais, i.e. les mots très courant sans grande valeur sémantique.
	"you" a été retiré de la liste des top words car les insultes en anglais l'utilisent souvent."""

	sw = open("stop_words.txt")
	s = sw.read()
	stop_words = s.split('\n')

	return stop_words


def load_bad_words():
	"""Retourne la liste des bad words établie par Google.

	SORTIE
	------------
	tab : liste de strings
		La liste des bad words sous la forme d'un tableau.
	"""
	bw = open("bad_words.txt", "r")
	s = bw.read()
	bad_tab = re.split("\W+", s)
	# le premier et dernier élément de la liste sont vides (conséquence du split, les premier et dernier caractères du fichier étant des guillemets)
	return bad_tab[1:-1]


def get_vocab(processed_corpus):
	"""Retourne la liste des mots employés dans tout le corpus.
	ENTREE
	---------
	processed_corpus : liste de strings
		Tableau de commentaires nettoyé, tokenizé, racinisé

	SORTIE
	---------
	vocab : liste de strings
		Le vocabulaire employé dans le corpus
	"""

	vocab = set()
	for comment in processed_corpus:
		vocab.update(comment)

	return np.array(list(vocab))


def get_cross_vocab(train, test):
	"""Fait en sorte d'avoir les mêmes features dans le train et dans le test."""

	train_vocab = set(get_vocab(train))
	test_vocab = set(get_vocab(test))

	return list(train_vocab.intersection(test_vocab))


def clean(corpus):
	"""Nettoie les commentaires et remplace les éléments indésirables par des balises adaptées.
	Largement inspiré de https://github.com/andreiolariu/kaggle-insults/blob/master/nlp_dict.py
	
	ENTREE
	----------
	corpus : liste de strings
		Le tableau de commentaires

	SORTIE
	----------
	tab : liste de strings
		Tableau de commentaire modifié
	"""

	# signes d'encodage et balises html
	regex_enc_tag = re.compile(r"\\r|\\n|\\x(\w{2})|\\u[A-F\d]{4}|<[^>]*>")
	# adresses url
	regex_url = re.compile(r"https?://[^\s]*")
	# limite le nombre de lettres identiques consécutives à 2
	regex_reps = re.compile(r"([a-zA-Z])\1\1+(\w*)")
	# nombres seuls, a priori pas indicatifs du caractère insultant
	regex_nbs = re.compile(r"\s\d+[.!?,;:\"\s]?")
	# dates, on peut penser que certaines dates sont importantes (comme 9/11) mais je pense qu'elles ne disent pas grand chose du caractère insultant
	regex_date = re.compile(r"\s\d?\d[./\-]\d?\d[./-]\d\d\d\d[.!?,;:\"\s]") # ne vérifie pas que les valeurs sont plausibles mais devrait suffire
	# recompose les mots dont les lettres sont séparées par des points ou des tirets
	regex_dotted_words = re.compile(r"[.\-\s](\w)[.-](\w)") # pour ne pas confondre les acronymes ou tentative de dissimulation de mots avec de la ponctutation
	# pour les personnes qui jurent comme dans une BD
	regex_curse = re.compile(r"([a-zA-Z]*)[#%&\*\$]{2,}([a-zA-Z]*)")
	# interprète les smileys
	regex_happy = re.compile(r"\^\^|[B8:;=xX]-?([)DpP]|\)\))")
	regex_sad = re.compile(r":'\(|[:=]-?[S(/$]")
	regex_ang = re.compile(r"-_-|:@|:-?\|")
	# gère les don't, can't etc..self.
	regex_neg = re.compile(r"(\w+)n't")
	# gère les you're
	regex_youare = re.compile("you're")
	# gère les 've
	regex_have = re.compile(r"(\w+)'ve")
	# sépare les mots reliés par un underscore ou un tiret
	regex_und = re.compile(r"(\w)+[_\-](\w)+")

	tab = []
	for comment in corpus:

		no_enc_tag = regex_enc_tag.sub(" ", comment)
		no_url = regex_url.sub(" _url ", no_enc_tag)
		no_nbs = regex_nbs.sub(" _nb ", no_url)
		no_date = regex_date.sub(" _date ", no_nbs)
		no_dotted_words = regex_dotted_words.sub(r" \1\2 ", no_date)
		no_reps = regex_reps.sub(r" \1\1\2 _rep ", no_dotted_words)
		no_curse = regex_curse.sub(" fuck ", no_reps) # remplacer par fuck ou par une balise spéciale ? fuck dans bad_words au moins
		no_happy = regex_happy.sub(" _happysmiley ", no_curse)
		no_sad = regex_sad.sub(" _sadsmiley ", no_happy)
		no_ang = regex_ang.sub(" _angrysmiley ", no_sad)
		no_neg = regex_neg.sub(r"\1 not", no_ang)
		youare = regex_youare.sub("you are", no_neg)
		have = regex_happy.sub(r"\1 have", youare)
		no_und = regex_und.sub(r"\1 \2", have)

		tab.append(no_und)

	return tab


def remove_stop_words_punctuation(tokenized_corpus):
	"""Retire les stop words et la ponctuation des commentaires.

	ENTREE
	------------
	tokenized_corpus : liste 2d de strings
		Le tableau des commentaires tokenizés.

	SORTIE
	-----------
	no_stop : liste 2d de strings
		Le même tableau sans les stop words.
	"""

	stopwords = load_stop_words()

	no_stop_punct = []
	for comment in tokenized_corpus:
		no_stop_punct.append([w.strip("\\") for w in comment if w not in stopwords and w not in string.punctuation and w != ""])

	return no_stop_punct


def count_punctuation(corpus, ratio=True, auto=True):
	"""Retourne des informations sur le nombre de signes de ponctuation dans chaque commentaire.
	
	ENTREE
	------------
	corpus : liste de strings
		Le tableau de commentaires (nettoyé normalement), tokenizé ou non selon la valeur de auto.

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	auto : booléen
		Si l'on a utilisé nltk dans tokenize. La ponctuation apparaît alors dans le tableau tokenizé.

	SORTIE
	------------
	cnt_tab : numpy.ndliste
		Tableau contenant pour chaque commentaire le nombre de signe de ponctuation, divisé ou non, selon la valeur de ratio,
		par la longueur du commentaire (en nombre de mots si auto=True, en nombre de caractères sinon).
	"""
	
	cnt_tab = []
	if auto:
		for comment in corpus:
			cnt = 0
			for word in comment:
				if word in string.punctuation:
					cnt += 1
			if ratio:
				cnt_tab.append(cnt/len(comment))
			else:
				cnt_tab.append(cnt)

	else:
		for comment in corpus:
			comment_no_space = [c for c in comment if c not in " _'"] #  on ajoute des espaces et _REP dans clean donc on ne les considère pas
			cnt = 0
			for c in comment_no_space:
				if c in string.punctuation:
					cnt += 1
			if ratio:
				cnt_tab.append(cnt/len(comment_no_space))
			else:
				cnt_tab.append(cnt)

	cnt_tab = np.array(cnt_tab)

	return cnt_tab


def count_capitals(tokenized_corpus, ratio=True):
	"""Compte le nombre absolu ou relatif de lettres majuscules dans le tableau de commentaires tokenizé.
	
	ENTREE
	------------
	tokenized_corpus : liste 2d de strings
		Le tableau des commentaires tokenizés

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	SORTIE
	------------
	tokenized_corpus_low : liste 2d de strings
		Copie de tokenized_corpus mais avec tous les mots en minuscules

	cnt_tab : numpy.ndliste
		Tableau contenant pour chaque commentaire le nombre de lettres en majuscule, divisé ou non, selon la valeur de ratio,
		par la longueur du commentaire.
	"""

	cnt_tab = []

	for tokenized_comment in tokenized_corpus:
		cnt_cap = 0
		cnt = 0
		for word in tokenized_comment:
			for c in word:
				if c == c.upper() and c not in string.punctuation:
					cnt_cap += 1
				cnt += 1

		if ratio:
			cnt_tab.append(cnt_cap/cnt)
		else:
			cnt_tab.append(cnt_cap)

	cnt_tab = np.array(cnt_tab)

	return cnt_tab


def get_bw_stats(tokenized_corpus, ratio=True):
	"""Retourne le ratio de bad_words contenus dans les commentaires

	ENTREE
	-------------
	tokenized_corpus : liste de strings
		Le tableau de commentaires tokenizé.

	ratio : booléen
		True si on veut une valeur relative, False si absolue.

	SORTIE
	-------------
	bw_cnt : numpy.ndliste
		Le tableau contenant le ratio de chaque commentaire.
	"""

	bad_words = load_bad_words()

	bw_cnt = []
	for comment in tokenized_corpus:
		comment_no_punct = [word for word in comment if word not in string.punctuation]
		cnt = 0
		for word in comment_no_punct:
			if word in bad_words:
				cnt += 1
		if ratio:
			#bw_cnt.append({'cnt_bw' : cnt, 'tot' : len(comment_no_punct)})
			bw_cnt.append(cnt/len(comment_no_punct))
		else:
			bw_cnt.append(cnt)

	bw_cnt = np.array(bw_cnt)

	return bw_cnt


def get_statistics(tokenized_corpus, ratio=True):
	"""Retourne les nombres ou ratio de :
	- signes de ponctuation ;
	- majuscules ;
	- bad words.
	A n'utiliser que si l'on a pris auto=True auparavant.
	
	ENTREE
	------------
	tokenized_corpus : liste 2d de strings
		Le tableau des commentaires tokenisés

	ratio : booléen
		Si on veut une valeur absolue ou relative à la taille du commentaire

	SORTIE
	------------
	tokenized_corpus_low = liste de strings
		Le tableau de commentaires tokenizé et tout en minuscules.

	maj_stats : numpy.ndliste
		Le tableau de statistiques sur les majuscules.

	punct_stats : numpy.ndliste
		Le tableau de statistiques sur la ponctuation.

	bw_stats : numpy.ndliste
		Le tableau de statistiques sur les bad words.
	"""

	tokenized_corpus_low, maj_stats = count_capitals(tokenized_corpus, ratio)
	punct_stats = count_punctuation(tokenized_corpus, ratio)
	bw_stats = get_bw_stats(tokenized_corpus, ratio)

	return tokenized_corpus_low, maj_stats, punct_stats, bw_stats