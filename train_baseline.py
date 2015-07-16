#import sklearn.mixture as sm
import sklearn.linear_model as sl
import numpy as np
import cPickle as pkl
import gzip
import sys

train, valid, test = pkl.load(gzip.open('mnist.pkl.gz', 'r'))

X = train[0]
y = train[1]

classifier = sl.LogisticRegression(penalty='l2', fit_intercept=True, solver='lbfgs', max_iter=1000, multi_class='multinomial', verbose=1)
classifier.fit(X, y)

pkl.dump(classifier, open('output_baseline/baseline.pkl', 'wb'))

X = test[0]
y = test[1]
print classifier.score(X, y)











