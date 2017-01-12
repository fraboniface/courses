import numpy as np
import optim

class LogisticRegression():

	def __init__(self, rho=1.0, tol=10^(-8), a=0.5, b=2.0, beta=0.5, gamma=0.5):

		self._rho = rho
		self._tol = tol
		self._a = a
		self._b = b
		self._beta = beta
		self._gamma = gamma
		self._coeff = None
		self._intercept = 0.0

	@property
	def C(self):
		return self._C

	@C.setter
	def C(self, c):
		self._C = c

	@property
	def coeff(self):
		return self._coeff

	@property
	def intercept(self):
		return self._intercept

	
	def fit(self, X, y):
		"""Fit le prédicteur"""

		print("Fitting...")
		n, p = X.shape

		self._features = X
		self._labels = 2*y - 1

		self._coeff = np.zeros(p)
		self._intercept = 1.

		w0, w = optim.armijos_descent(self.compute_cost, self.compute_grad, self._intercept, self._coeff, self._a, self._b, self._beta, self._tol)
		#w0, w = optim.gradient_descent(self.compute_grad, self._intercept, self._coeff, self._gamma, self._tol)

		self._coeff = w
		self._intercept = w0

		return self

	def compute_cost(self, w0, w):
		"""Calcule la fonction de coût de la regression logistique."""

		X = self._features
		y = self._labels

		n, p = X.shape

		return np.sum(np.log(1+np.exp(-y*(np.dot(X,w)+w0))))/n  + (self._rho/2)*(np.linalg.norm(w)**2)


	def compute_grad(self, w0, w):
		"""Calcule le gradient de la fonction de coût."""

		X = self._features
		y = self._labels

		n, p = X.shape

		v = y / (1.+np.exp(y*(np.dot(X,w)+w0)))

		grad_w0 = -np.sum(v)/n
		grad_w = -np.dot(v,X)/n + self._rho * w

		return np.hstack((grad_w0,grad_w))

	def predict(self, X):
		"""Prédit la valeur cible en fonction de X."""

		values = 1. / (1. + np.exp(- X.dot(self._coeff) -  self._intercept))
		#print(values.min(), values.max())
		#print("Moyenne :", values.mean())

		return np.array((values > 0.5).astype(np.int))

	def score(self, X, y):
		"""Calcule le score en prédiction sur X et y."""

		return np.mean(self.predict(X) == y)