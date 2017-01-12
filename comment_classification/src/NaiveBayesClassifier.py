import numpy as np
import utils

class NBC():

	def __init__(self, vocab = None, mu=0, alpha=1):
		"""Constructeur de la classe NBC.

		ENTREE
		---------
		mu : float
			Le décalage que l'on veut appliquer à la log-vraisemblance. Utile seulement si alpha différent de 1.

		alpha : float
			La puissance à laquelle on veut élever la vraisemblance, qui multipliera donc la log-vraisemblance.
		"""

		self._mu = mu
		self._alpha = alpha
		self._dic = {}
		self._imbalance = 0.5
		self._vocab = vocab


	@property
	def mu(self):
		return self._mu
	
	@mu.setter
	def mu(self, val):
		self._mu = val


	@property
	def alpha(self):
		return self._alpha
	
	@alpha.setter
	def alpha(self, val):
		self._alpha = val

	@property
	def dic(self):
		return self._dic


	def fit(self,X,y):
		"""Applique le classifieur aux données.

		ENTREE
		---------
		X : liste de strings
			Le tableau de commentaires passé par la fonction process

		y : liste de ints
			Les labels qui vont avec X.
		"""

		print("Fitting...")

		self._imbalance = y.mean()

		n = len(X)
		if self._vocab is None:
			self._vocab = utils.get_vocab(X)

		for word in self._vocab:
			cnt0 = 0
			cnt1 = 0
			for i,com in enumerate(X):
				if type(word).__name__ == 'tuple':
					tuples = [(com[i],com[i+1]) for i in range(len(com)-1)]
					if word in tuples:
						if y[i] == 0:
							cnt0 += 1
						else:
							cnt1 += 1
				else:
					if word in com:
						if y[i] == 0:
							cnt0 += 1
						else:
							cnt1 += 1

			#print(cnt1)
			#print(cnt0)
			#print("Putting", np.log(cnt1/(cnt0+cnt1)*(1-self._mu))*self._alpha, "in", word)
			self._dic[word] = np.log(cnt1/(cnt0+cnt1)*(1-self._mu))*self._alpha

		return self


	def predict(self, test):
		"""Prédit les classes pour le corpus de test."""

		predictions = []
		neutral = np.log(self._imbalance - self._mu)*self._alpha

		for com in test:
			tab = []
			for w in com:
				if w in self._dic.keys():
					tab.append(self._dic[w])
				else:
					tab.append(neutral)

			val = np.mean(tab)
			predictions.append(int(val > neutral))

		return np.array(predictions)


	def score(self, X, y):
		"""Calcule le score sur X et y."""
		return np.mean(self.predict(X)==y)