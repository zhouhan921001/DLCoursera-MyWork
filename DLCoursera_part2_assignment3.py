import numpy as np 

def random_mini_batch(X,Y,seed):
	"""
	Randomize the trainning dataset
	"""

	np.random.seed(seed)
	m = X.shape[1]
	mini_batch = []

	# Step1: shuffle
	permutation = list(np.random.permutation(m))
	shuffle_X = X[:,permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))

	#Step2: partition
	num_intergle_minibatch = math.floor(m / mini_batch_size)
	for i in range(num_intergle_minibatch):
		mini_batch_X = shuffle_X[:,i * mini_batch_size : (i+1) * mini_batch_size]
		mini_batch_Y = shuffle_Y[:,i * mini_batch_size : (i+1) * mini_batch_size]
		mini_batch = (mini_batch_X,mini_batch_Y)
		mini_batchs.append(mini_batch)

	if m % mini_batch_size != 0:
		mini_batch_X = shuffle_X[:,num_intergle_minibatch * mini_batch_size :]
		mini_batch_Y = shuffle_Y[:,num_intergle_minibatch * mini_batch_size :]
		mini_batch = (mini_batch_X,mini_batch_Y)
		mini_batchs.append(mini_batch)

	return mini_batchs

def initialize_velocity(params):
	"""
	"""

	L = len(params) // 2
	v = {}

	for l in range(L):
		v['dw'+str(l+1)] = np.zeros_like(params['w'+str(l+1)])
		v['db'+str(l+1)] = np.zeros_like(params['b'+str(l+1)])

	return v

def momentum(params,v,beta,grads,learning_rate):
	"""
	Update parameters with momentum
	"""

	L = len(params) // 2

	for l in range(L):
		v['dw'+str(l+1)] = beta * v['dw'+str(l+1)] + (1 - beta) * grads['dw'+str(l+1)]
		v['db'+str(l+1)] = beta * v['db'+str(l+1)] + (1 - beta) * grads['db'+str(l+1)]

		params['w'+str(l+1)] = params['w'+str(l+1)] - learning_rate * v['dw'+str(l+1)]
		params['b'+str(l+1)] = params['b'+str(l+1)] - learning_rate * v['db'+str(l+1)]

	return params,v

def initialize_adam(params):
	"""
	"""

	L = len(params) // 2
	v = {}
	s = {}

	for l in range(L):
		v['dw'+str(l+1)] = np.zeros_like(params['w'+str(l+1)])
		v['db'+str(l+1)] = np.zeros_like(params['b'+str(l+1)])
		s['dw'+str(l+1)] = np.zeros_like(params['w'+str(l+1)])
		s['db'+str(l+1)] = np.zeros_like(params['b'+str(l+1)])

	return v,s

def adam(params,v,beta,grads,learning_rate,epsilon):
	"""
	Update parameters with momentum
	"""

	L = len(params) // 2

	for l in range(L):
		v['dw'+str(l+1)] = beta1 * v['dw'+str(l+1)] + (1 - beta1) * grads['dw'+str(l+1)]
		v['db'+str(l+1)] = beta1 * v['db'+str(l+1)] + (1 - beta1) * grads['db'+str(l+1)]
		s['dw'+str(l+1)] = beta2 * s['dw'+str(l+1)] + (1 - beta2) * np.power(grads['dw'+str(l+1)],2)
		s['db'+str(l+1)] = beta2 * s['db'+str(l+1)] + (1 - beta2) * np.power(grads['dw'+str(l+1)],2)

		v_correct['dw'+str(l+1)] = v["dw" + str(l + 1)] / (1 - np.power(beta1, t))
		v_correct['db'+str(l+1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
		s_correct['dw'+str(l+1)] = s["dw" + str(l + 1)] / (1 - np.power(beta2, t))
		s_correct['db'+str(l+1)] = s["db" + str(l + 1)] / (1 - np.power(beta2, t))

		params['w'+str(l+1)] = params['w'+str(l+1)] - learning_rate * v['dw'+str(l+1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)
		params['b'+str(l+1)] = params['b'+str(l+1)] - learning_rate * v['db'+str(l+1)] / np.sqrt(s["dW" + str(l + 1)] + epsilon)

	return params,v,s
