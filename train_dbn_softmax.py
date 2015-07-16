#import sklearn.mixture as sm
import sklearn.linear_model as sl
import numpy as np
import cPickle as pkl
import gzip
import sys

train, valid, test = pkl.load(gzip.open('mnist.pkl.gz', 'r'))

all_params = pkl.load(open('output_dbn/pre_trained_dbn_layers.pkl', 'rb'))

def sigmoid(z):
	s = 1.0 / (1.0 + np.exp(-1.0 * z))
	return s

X = train[0]
y = train[1]

X_0 = sigmoid(np.dot(X, all_params[0][0])+all_params[0][1])
X_1 = sigmoid(np.dot(X_0, all_params[1][0])+all_params[1][1])
X_2 = sigmoid(np.dot(X_1, all_params[2][0])+all_params[2][1])

classifier0 = sl.LogisticRegression(penalty='l2', fit_intercept=True, solver='lbfgs', max_iter=1000, multi_class='multinomial', verbose=1)
classifier0.fit(X_0, y)
pkl.dump(classifier0, open('output_dbn/dbn_softmax0.pkl', 'wb'))

classifier1 = sl.LogisticRegression(penalty='l2', fit_intercept=True, solver='lbfgs', max_iter=1000, multi_class='multinomial', verbose=1)
classifier1.fit(X_1, y)
pkl.dump(classifier1, open('output_dbn/dbn_softmax1.pkl', 'wb'))

classifier2 = sl.LogisticRegression(penalty='l2', fit_intercept=True, solver='lbfgs', max_iter=1000, multi_class='multinomial', verbose=1)
classifier2.fit(X_2, y)
pkl.dump(classifier2, open('output_dbn/dbn_softmax2.pkl', 'wb'))

X = test[0]
X_0 = sigmoid(np.dot(X, all_params[0][0])+all_params[0][1])
X_1 = sigmoid(np.dot(X_0, all_params[1][0])+all_params[1][1])
X_2 = sigmoid(np.dot(X_1, all_params[2][0])+all_params[2][1])
y = test[1]

print classifier0.score(X_0, y)
print classifier1.score(X_1, y)
print classifier2.score(X_2, y)