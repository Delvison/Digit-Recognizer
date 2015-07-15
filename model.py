import os
import sys
from theano import *
import theano.tensor as T
import numpy as np
import cPickle as cp

def initialize(finetune_lr=0.1, pretraining_epochs=100,
             pretrain_lr=0.01, k=1, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=10):
    # get training data
    training_set = get_training_set()

    # TODO: get test set

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
    p_y_given_x = T.nnet.softmax(T.dot(x, W) + b)

    # how to compute prediction as class whose probability is maximal
    y_prediction = T.argmax( matrix_given_x, axis=1 )

    # gather parameters for gradient calculation
    parameters = [W, b]

    # vector for cost calculation
    y = T.ivector('y')

    # calculate cost using negative log likelihood
    cost = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # compute the gradient of cost with respect to theta = (W,b)
    grad_W = T.grad(cost=cost, wrt=W)
    grad_b = T.grad(cost=cost, wrt=b)

    # specify how to update the params of the model as a list of
    # (variable, update expression)
    updates = [(W, W - learning_rate * grad_W),
                (b, b - learning_rate * grad_b)]

    index = T.lscalar('index')
    batch_size = 600

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print "...training model"



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
