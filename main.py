import os
import sys
import timeit
import cPickle
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from DBN import *

if not os.path.isfile('pre_trained_dbn.pkl'):
    print " ...DBN does not exist yet."
    pre_trained_dbn = test_DBN()
    # save the the pretrained dbn model
    with open('pre_trained_dbn.pkl', 'w') as f:
        cPickle.dump(classifier, f)
else:
    print "...DBN exists already"

def get_rbm_vals(dbn):
    results = []
    rbms = rbm_layers
    for i in rbms:
        layer_data = [rbms[i].W, rbms[i].hbias, rbms[i].vbias]
        results.append(layer_data)
    return results
