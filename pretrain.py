"""
# This script generates a Deep Belief Network using the MNIST data set then
# saves the DBN object and a list of all of its layers' weights in pickle files.
# It uses existing code from theano to do so.
#
# Theano code can be found in the following URL:
# http://deeplearning.net/tutorial/code/
#
"""

import os
import sys
import timeit
import cPickle
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from DBN import *

def pretrain_dbn():
		"""
		# Pretrains a DBN under MNIST data set
		# @author Delvison Castillo
		"""
		if not os.path.isfile('pre_trained_dbn.pkl'):
				print " ...DBN does not exist yet."
				pre_trained_dbn = test_DBN() # DBN object

				# save the the pretrained dbn model
				with open('pre_trained_dbn.pkl', 'w') as f:
						cPickle.dump(pre_trained_dbn, f)

				return pre_trained_dbn

		else:
				print "...DBN exists already"
				pre_trained_dbn = cPickle.load( open("pre_trained_dbn.pkl","rb") )
				return pre_trained_dbn

def get_rbm_vals(dbn):
		"""
		# Takes in a DBN object and returns a list of the layers' W, hbias, vbias
		# @author Delvison Castillo
		"""
		if not os.path.isfile('pre_trained_dbn_layers.pkl'):
				layers_data = []
				rbms = dbn.rbm_layers
				for i in xrange(dbn.n_layers):
						layer_data = [rbms[i].W.get_value(), rbms[i].hbias.get_value(),
						rbms[i].vbias.get_value()]
						layers_data.append(layer_data)

				# save the the layer data
				with open('pre_trained_dbn_layers.pkl', 'w') as f:
						cPickle.dump(layers_data, f)
				return layers_data
		else:
				layers_data = cPickle.load( open ("pre_trained_dbn_layers.pkl"))
				return layers_data


if __name__ == '__main__':
		pre_trained_dbn = pretrain_dbn()
		layers_data = get_rbm_vals(pre_trained_dbn)
