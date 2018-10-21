import numpy
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X,Y = load_planar_dataset()

num_train = X.shape[1]
X_shape = X.shape
Y_shape = Y.shape

def model_size(X,Y):

	n_x = X.shape[0]  ## size of input layer
	#n_h = 4
	n_y = Y.shape[0]  ## size of output layer

	return (n_x,n_h,n_y)

def initilze(n_x,n_h,n_y):

	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.random.randn(n_h,1)
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.random.randn(n_y,1)

	parameter = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}

	return parameter

def forward_propagate(param,X):

	W1 = param['W1']
	b1 = param['b1']
	W2 = param['W2']
	b1 = param['b1']

	Z1 = np.dot(W1,X) + b1
	A1 = sigmoid(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

	return A2,cache

def compute_cost(A2,Y):

	m = Y.shape[1]

	logprob = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)

	cost = - np.sum(logprob)/m

	return cost

def backward_propagete(param,cache,X,Y):

	W1 = param['W1']
	W2 = param['W2']

	Z2 = cache['Z2']
	A2 = cache['A2']
	Z1 = cache['Z1']
	A1 = cache['A1']

	dZ2 = A2 - Y
	dW2 = 1/m * np.dot(dZ2,A1.T)
	db2 = 1/m * np.sum(dZ2,axis=1,keepdims=True)
	dZ1 = np.multiply(np.dot(W2.T,dZ2),1 - np.power(A1,2))
	dW1 = 1/m * np.dot(dZ1,X.T)
	db1 = 1/m * np.sum(dZ1,axis=1,keepdims=True)

	grads = {"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}

	return grads

def update_param(param,grads,learning_rate=1.2):

	W1 = param['W1']
	b1 = param['b1']
	W2 = param['W2']
	b2 = param['b2']

	dW1 = grads['W1']
	db1 = grads['b1']
	dW2 = grads['W2']
	db2 = grads['b2']

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameter = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}

	return parameter

def nn_model(X,Y,n_h,num_iter):

	n_x = model_size(X,Y)[0]
	n_y = model_size(X,Y)[2]

	param = initilze(n_x,n_h,n_y)
	W1 = param['W1']
	b1 = param['b1']
	W2 = param['W2']
	b2 = param['b2']

	for i in range(num_iter):

		A2,cache = forward_propagate(param,X)
		cost = compute_cost(A2,Y)
		grads = backward_propagete(param,cache,X,Y)
		param = update_param(param,grads)

	return param
