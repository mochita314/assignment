#!/usr/bin/python

import numpy as np
import math
import random
import argparse

# build a neural network

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

    
        


        
