import os
import sys
from theano import *
import theano.tensor as T
import numpy as np
import cPickle as cp

'''

'''
def initialize():
    # initialize the W weight matrix to 0's
    W = theano.shared(
        value= np.zeros(
            (784, 10),
            dtype=theano.config.floatX
        ),
        name = 'W',
        borrow = True
    )

    # initialize biases vector to 0's
    b = theano.shared(
        np.zeros(
            (10,),
            dtype=theano.config.floatX
        ),
        name = 'b',
        borrow = True
    )

    # calculate softmax
    matrix_given_x = T.nnet.softmax(T.dot(T.dmatrix('x'), W) + b)

    # cost function for maximum likelihood with L2 regularization
    y_prediction = T.argmax( matrix_given_x, axis=1 )

    # gather parameters for gradient calculation
    parameters = [W, b]

    # calculate gradient tensor.grad(inputs, output_grads)
    gradients = T.grad(cost, )

    # sgd as grads
    (f_grad_shared, f_update) = sgd()



def train():
    """
    # Function to train the neural network
    """
    # TODO: WRITE ME
    return 0


def sgd(learn_rate, params_list, grads, x, y, cost):
    """
    # Function for stochastic gradient descent
    """
    # TODO: WRITE ME
    # for param,grad in zip(params, grads):
    return 0


def predict():
    """
    # Function to predict a classification
    """
    # TODO: WRITE ME
    return 0


def get_training_set():
    """
    # Reads data from csv file and returns a list of two lists where the first
    # list is the tags for the corresponding image in the second list.
    """
    training_file = 'train-1.csv'
    tags = []
    pixels = []
    #open training file
    with open(training_file) as f:
        # read training file line by line
        for sequence in f:
            temp = sequence.split(",")
            tags.append( temp.pop(0))
            pixels.append(temp)
    return (tags, pixels)


def conv_output(output):
    """
    # Returns the equivalent number to the given binary number that is an output
    # from the network
    """
    nums = {1000000000: 1,
            0100000000: 2,
            0010000000: 3,
            0001000000: 4,
            0000100000: 5,
            0000010000: 6,
            0000001000: 7,
            0000000100: 8,
            0000000010: 9,
            0000000001: 0}
    return nums[output]

if __name__ == '__main__':
    initialize()
