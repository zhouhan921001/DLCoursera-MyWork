import numpy as np
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward

def initialize_two_layer(n_x,n_h,n_y):

	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros(n_h,1)
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros(n_y,1)

	param = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}

	return param

def initialize_l_layer(layer_dims):
	
	param = {}
	L = len(layer_dims)

	for l in range(1, L):
		param['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
		param['b' + str(l)] = np.zeros(layer_dims[l],1)

	return param

def linear_forward(W,A,b):
	"""
	Implement the linear part of neural unit
	"""

	Z = np.dot(W,A) + b

	return Z

def linear_activation_forward(A_pre,W,b,activation):
	"""
	Implement neural unit with the activation of Relu or sigmoid
	"""

	if activation == "Relu":

		Z = linear_forward(W,A_pre,b)
		A,activation_cache = relu(Z)

	elif activation == "sigmoid":

		Z = linear_forward(W,A_pre,b)
		A,activation_cache = sigmoid(Z)

		backward_used_cache = (A_pre,W,b)
		cache = (backward_used_cache,activation_cache)
	return A,cache

def L_model_forward(X,param):
	"""
	Implement forward propagation for L layers model
	"""

	caches = []
	L = len(param) // 2
	A = X

	for l in range(1,L):

		A,cache = linear_activation_forward(A,param['W'+str(l)],param['b'+str(l)],Relu)
		caches.append(cache)

	Al,cache = linear_activation_forward(A,param['W'+str(l)],param['b'+str(l)],Relu)
	caches.append(cache)

	return Al,caches

def linear_backward(dz,cache):
	"""
	Implement the backward propagation of linear part
	"""

	m = dz.shape[1]
	dw = np.dot(dz,cache[0]) / m
	db = np.sum(dz) / m
	dA_pre = np.dot(cache[1],dz)

	return dw,db,dA_pre

def linear_activation_backward(dA,cache,activation):
	"""
	Implement the backward propagation of neural unit
	"""

	if activation == "Relu":
		dz = relu_backward(dA,cache[1])

	elif activation == "sigmoid":
		dz = sigmoid_backward(dA,cache[1])

	dw,db,dA_pre = linear_backward(dz,cache[0])

	return dw,db,dA_pre

def L_model_backward(AL,Y,caches):
	"""
	Implement the backward propagation for L layer model
	"""
	grads = {}
	L = len(caches)

	dAl = - (np.divide(Y,AL) - np.divide(1-Y,1-AL))
	grads['dw'+str(L)],grads['db'+str(L)],grads['dA'+str(L)] = linear_activation_backward(dAL,caches[-1],"sigmoid")

	for l in reversed(range(L-1)):
		cache = caches[l]
		grads['dw'+str(l+1)],grads['db'+str(l+1)],grads['dA'+str(l+1)] = linear_activation_backward(grads['dA'+str(l+2)],
																									cache,"Relu")
	return grads

def update_param(param,grads,learning_rate):
	"""
	Update the parameters
	"""

	L = len(param) // 2
	for l in range(L):
		param['W'+str(l+1)] = param['W'+str(l+1)] - learning_rate * grads['W'+str(l+1)]
		param['b'+str(l+1)] = param['b'+str(l+1)] - learning_rate * grads['b'+str(l+1)]

	return param
