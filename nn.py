#!/usr/bin/python

import numpy as np
import math
import random
import argparse

# Implement activation function layers

class Sigmoid:

    def __init__(self):
        self.x = None
        self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = 1 / (1 + np.exp(-self.x))
        self.y = y
        return y
    
    # backword propagation
    def backward(self,dout):
        return dout*self.y*(1-self.y)

class Relu:

    def __init__(self):
        self.x = None
        self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = x*(x>0)
        self.y = y
        return y
    
    # backward propagation
    def backward(self,dout):
        return dout*(self.x>0)

class Softmax:

    def __init__(self):
        self.x = None
        self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        exp_x = np.exp(x-x.max(axis=1,keepdims=True)) # prevent overflow
        sum_exp_x = np.sum(exp_x,axis=1,keepdims=True)
        y = exp_x / sum_exp_x
        self.y = y
        return y

# Implement linear layer

class Affine:

    def __init__(self):
        # initialize
        self.x = None
        self.y = None
        self.param = True
        self.params = {}
        self.grads = {}
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = np.dot(x,self.params['W']) + self.params['b']
        self.y = y
        return y
    
    # backward propagation
    def backward(self):
        self.grads['dW'] = 
        self.grads['db'] = 


class SGD:

    def __init__(self,lr):
        self.lr = lr
        self.network = None
    
    def setup(self,network):
        self.network = network
    
    def update(self):
        for layer in self.network.layers:
            if layer.param: # don't apply for actibavtion function layers
                layer.params['W'] -= self.lr * layer.grads['dW']
                layer.params['b'] -= self.lr * layer.grads['db']

# Implement Multilayer perceptron

class MLP(layer):

    def __init__(self):


