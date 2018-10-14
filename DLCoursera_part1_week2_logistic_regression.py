import numpy as np

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

#reshape data
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T


def sigmoid(z):

	sig = 1 / (1 + np.exp(-z))

	return sig

def initialize(dim):

	w = np.zeros(shape=(dim,1))
	b = 0

	return w,b

def propagate(w,b,X,Y):

	m = X.shape[1]
	# forward propagate
	z = np.dot(w.T,X) + b
	A = sigmoid(z)
	#cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

	# backward propagate
	dw = 1/m * np.dot(X,(A - Y).T)
	db = 1/m * np.sum(A - Y)

	return dw,db

def optimize(X,Y,w,b,num_iter,learning_rate):

	for i in range(num_iter):
		dw,db = propagate(w,b,X,Y)

		w = w - learning_rate * dw
		b = b - learning_rate * db

	return w,b

def predict(w,b,X):

	z = np.dot(w.T,X) + b
	A = sigmoid(z)

	for i in range(A.shape[1]):
		Y_predict = 1 if A[0,i] > 0.5 else 0

	return Y_predict

def model(X_train,Y_train,X_test,Y_test,num_iter,learning_rate):

	w,b = initialize(train_set_x.shape[0])
	
	w,b = optimize(X_train,Y_train,w,b,num_iter,learning_rate)

	Y_predict = predict(w,b,X_test)

	Y_predict_train = predict(w,b,X_train)

	print("test_accuracy: {} %".format(np.mean(np.abs(Y_test - Y_predict)) * 100))
	print("train_accuracy: {} %".format(np.mean(np.abs(Y_train - Y_predict_train)) * 100))


