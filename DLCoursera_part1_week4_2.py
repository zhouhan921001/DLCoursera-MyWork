import numpy as np
import h5py

train_x_orig,train_y,test_x_orig,test_y,classes = load_data()
m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

train_x_flatten = train_x_orig.reshape(m_train,-1).T
test_x_flatten = test_x_orig.reshape(m_test,-1).T 

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

n_x = 12288
n_h = 7
n_y = 1
layer_dims = (n_x,n_h,n_y)

def two_layer_model(X,Y,layer_dims,num_iter,learning_rate):
	"""
	Implement a two layers model and train it
	"""

	grads = {}
	(n_x,n_h,n_y) = layer_dims

	param = initialize_two_layer(n_x,n_h,n_y)

	for i in range(num_iter):

		A1,cache1 = linear_activation_forward(X,param['W1'],param['b1'],"Relu")
		A2,cache2 = linear_activation_forward(A1,param['W2'],param['b2'],"sigmoid")

		dA2 = - (np.divide(Y,A2) - np.divide(1-Y,1-A2))

		dw2,db2,dA1 = linear_activation_backward(dA2,cache2,"sigmoid")
		dw1,db1,dA0 = linear_activation_backward(dA1,cache1,"Relu")

		grads['dw2'] = dw2
		grads['db2'] = db2
		grads['dw1'] = dw1
		grads['db1'] = db1

		param = update_param(param,grads,learning_rate)

	return param

layer_dims = [12288,20,7,5,1]

def L_layer_model(X,Y,layer_dims,num_iter,learning_rate):
	"""
	Implement a L layers model and train it
	"""
	param = initialize_l_layer(layer_dims)

	for i in range(num_iter):
		Al,caches = L_model_forward(X,param)
		grads = L_model_backward(Al,Y,caches)
		param = update_param(param,grads,learning_rate)

	return param
