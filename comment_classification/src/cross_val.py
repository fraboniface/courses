import numpy as np

def cv_score(clf,X,y,cv=3):

	X_splits = np.array_split(X,cv)
	y_splits = np.array_split(y,cv)

	scores = []
	for i in range(cv):
		X_train = np.vstack(np.delete(X_splits,i,0))
		y_train = np.hstack(np.delete(y_splits,i,0))
		
		X_test, y_test = X_splits[i], y_splits[i]
		
		clf.fit(X_train, y_train)
		scores.append(clf.score(X_test,y_test))

	scores = np.array(scores)

	return scores