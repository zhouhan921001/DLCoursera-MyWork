###################################################
# This .py file about the method of initialization 
# 1) zeros initialization
# 2) random initialization
# 3) He initialization
###################################################

import numpy as np 

def zeros_initialization(layers_dims):
	L = len(layers_dims)
	param = {}

	for l in range(1,L):
		param['W'+str(l)] = np.random.zeros(layers_dims[l],layers_dims[l-1])
		param['b'+str(l)] = np.random.zeros(layers_dims[l],1)

	return param

def random_initialization(layers_dims):
	L = layers_dims
	param = {}

	for l in range(1,L):
		param['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * 10
		param['b'+str(l)] = np.random.zeros(layers_dims[l],1)

	return param

def He_initialization(layers_dims):
	L = layers_dims
	param = {}

	for l in range(1,L):
		param['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
		param['b'+str(l)] = np.random.zeros(layers_dims[l],1)

	return param
