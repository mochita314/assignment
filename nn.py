#!/usr/bin/python

import numpy as np
import math
import random

# activation function layers

class Sigmoid():

    def __init__(self):
        self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    # backword propagation
    def backward(self,dout):
        return dout*self.y*(1-self.y)

class ReLU():

    def __init__(self):
        self.name = "ReLU"
        self.x = None
        #self.y = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        return x*(x>0)
    
    # backward propagation
    def backward(self,dout):
        return dout*(self.x>0)

class Softmax():

    # the way of implementation of backward propagation is different from other layers
    # since this layer will be used as the last one

    def __init__(self):
        self.x = None
        self.y = None
        #self.t = None
        self.param = False
    
    # forward propagation
    def __call__(self,x):
        self.x = x
        exp_x = np.exp(x-x.max(axis=1,keepdims=True)) # prevent overflow
        sum_exp_x = np.sum(exp_x,axis=1,keepdims=True)
        y = exp_x / sum_exp_x
        self.y = y
        return y
    
    # backward propagation
    """
    def backward(self,dout=1):
        dout = (self.y-self.t) / len(self.x)
        return dout
    """

# linear layer
class Affine():

    def __init__(self,input_dim,output_dim):
        self.name = "Affine"
        self.x = None
        self.y = None
        self.param = True
        #self.params = {}
        self.dW = None
        self.db = None
        # initialize weight matrix
        std = np.sqrt(2.0 / input_dim)
        self.W = std * np.random.randn(input_dim,output_dim)
        # initialize bias vector
        self.b= np.zeros(output_dim)

    
    # forward propagation
    def __call__(self,x):
        self.x = x
        y = np.dot(x,self.W) + self.b
        self.y = y
        return y
    
    # backward propagation
    def backward(self,delta):
        dout = np.dot(delta,self.W.T)
        self.dW = np.dot(self.x.T,delta)
        #print('delta:{}'.format(delta))
        self.db = np.dot(np.ones(len(self.x)), delta) 
        return dout

# optimizer
class SGD():

    def __init__(self,lr=0.01):
        self.lr = lr
        self.network = None
    
    def setup(self,network):
        self.network = network
    
    def update(self):
        for layer in self.network.layers:
            if layer.param: # don't apply for activation function layers
                #p = layer.W
                layer.W -= self.lr * layer.dW               
                layer.b -= self.lr * layer.db
                #print(layer.params['W'][:2])
                #print(self.lr)
                #print(layer.dW)


# Multilayer perceptron
class MLP():

    def __init__(self,layers=[]):
        self.layers = layers
        self.t = None
    
    def add_layer(self,layer):
        self.layers.append(layer)
    
    def forward(self,x,t):
        self.t = t
        #self.x = x
        self.y = x
        for layer in self.layers:
            #print(layer.name)
            self.y = layer(self.y)
        self.loss = np.sum(-t*np.log(self.y + 1e-7))/len(x)
        return self.loss
    
    def backward(self):
        dout = (self.y - self.t) / len(self.layers[-1].x)
        for layer in self.layers[-2::-1]:
            dout = layer.backward(dout)