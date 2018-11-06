import numpy as np 

def compute_cost_with_L2_reg(cost,params,lambd,m):
	"""
	Implement cost computation with L2 reguration
	"""

	W1 = params['W1']
	W2 = params['W2']
	W3 = params['W3']

	weight_sum = np.sum(np.square(W1)) + np.sum(np.square(W2)) +np.sum(np.square(W3))

	L2_reg_cost = lambd * weight_sum / (2 * m)

	return cost + L2_reg_cost


def linear_backward_with_L2_reg(dz,cache,lambd):
	"""
	Implement the backward propagation of linear part

	Argument:
	cache -- the cache of weight and bias of ith layer
	"""

	m = dz.shape[1]
	dw = np.dot(dz,cache[0]) / m + lambd * cache[1] / m
	db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
	dA_pre = np.dot(cache[1],dz)

	return dw,db,dA_pre

def linear_activation_backward_with_L2_reg(dA,cache,activation,lambd):
	"""
	Implement the backward propagation of neural unit
	"""

	if activation == "Relu":
		dz = relu_backward(dA,cache[1])

	elif activation == "sigmoid":
		dz = sigmoid_backward(dA,cache[1])

	dw,db,dA_pre = linear_backward_with_L2_reg(dz,cache[0],lambd)

	return dw,db,dA_pre

def backward_propagation_with_L2_reg(dal,caches,lambd,L):
	"""
	Implement backward proragation with L2 reguration

	Arguments:
	caches -- 
	"""

	grads = []

	dAL = - (np.divide(Y,AL) - np.divide(1-Y,1-AL))
	grads['dw'+str(L)],grads['db'+str(L)],grads['dA'+str(L-1)] = linear_activation_backward(dAL,caches[-1],"sigmoid",lambd)
	for l in reversed(range(L)):
		cache = caches[l]
		grads['dw'+str(l)],grads['db'+str(l)],grads['dA'+str(l-1)] = linear_activation_backward_with_L2_reg(grads['dA'+str(l)],
																									cache,"Relu",lambd)

	return grads

def forward_propag_with_dropout(params,X,keep_pro):
	"""
	Implement forward proragation with dropout
	"""

	caches = []
	A_pre = X
	L = len(params) // 2
	for l in range(L-1):
		Z = np.dot(params['W'+str(l+1)],A_pre)
		A_pre = relu(Z)
		D = np.random.rand(A_pre.shape[0],A_pre.shape[1])
		D = D < keep_pro
		A_pre = A_pre * D
		A_pre = A_pre / keep_pro
		cache = (A_pre,params['W'+str(l+1)],params['b'+str(l+1)],D)
		forward_cache = (cache,Z)
		caches.append(forward_cache)

	Z = np.dot(params['W'+str(L)],A_pre)
	Al = sigmoid(Z)
	cache = (A_pre,params['W'+str(L)],params['b'+str(L)])
	forward_cache = (cache,Z)
	caches.append(forward_cache)

	return Al,caches

def linear_backward_with_L2_reg(dz,cache,keep_pro):
	"""
	Implement the backward propagation of linear part

	Argument:
	cache -- the cache of weight and bias of ith layer
	"""

	m = dz.shape[1]
	dw = np.dot(dz,cache[0]) / m
	db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
	dA_pre = np.dot(cache[1],dz) / keep_pro

	return dw,db,dA_pre

def linear_activation_backward_with_L2_reg(dA,cache,activation,keep_pro):
	"""
	Implement the backward propagation of neural unit
	"""

	dA = dA * cache[0][3]
	if activation == "Relu":
		dz = relu_backward(dA,cache[1])

	elif activation == "sigmoid":
		dz = sigmoid_backward(dA,cache[1])

	dw,db,dA_pre = linear_backward_with_L2_reg(dz,cache[0],keep_pro)

	return dw,db,dA_pre

def backward_propag_with_dropout(caches,keep_pro,Y,Al):
	"""
	Implement backward proragation with dropout
	"""

	grads = {}
	L = len(caches)

	dAL = - (np.divide(Y,Al) - np.divide(1-Y,1-Al))
	grads['dw'+str(L)],grads['db'+str(L)],grads['dA'+str(L-1)] = linear_activation_backward(dAL,caches[-1],"sigmoid",keep_pro)

	for l in reversed(range(L)):
		cache = caches[l]
		grads['dw'+str(l)],grads['db'+str(l)],grads['dA'+str(l-1)] = linear_activation_backward(grads['dA'+str(l)],caches[l],"Relu",keep_pro)

	return grads
