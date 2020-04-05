from readimg import *
from random import random, shuffle
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

np.random.seed(0)

def initNetwork(sizes):
	W = []
	for i in range(len(sizes)-1):
		W.append(np.matrix([[random()*2-1 for y in range(sizes[i])] for x in range(sizes[i+1])]))

	return W

#Activation functions
def sigmoid(x, deriv = False):
	fx = expit(x)
	if not deriv:
		return fx

	return np.multiply(fx, (1 - fx))

#doesn't work unfortunalely
def relu(x, deriv = False):
	if deriv:
		x[x<=0] = 0
		x[x>0] = 1
		return x
	else:
		x[x<=0] = 0
# maps from a label to a vector containing expected outputs for each neuron.
def labelToExpected(label, size):
	target = np.zeros(size)
	if label[1] >= 31 and label[1] <= 57:
		i = label[1] - 21	
	else:
		i = label[1] - 70
	target[i] = 1
	return target

def forwardProp(W, x, activationfunc):
	layers = [x]
	x = x.T # x should be a vector (x by 1)

	for w in W:
		out = w.dot(x)
		layers.append(out.T)
		out = activationfunc(out)
		x = out;

	return layers, x.T;

def train(W, X, Y, activationfunc, learningrate = 1):
	#prediction matrix, keeps track of al intermediate layer outputs.
	layers = np.zeros((len(W)+1,Y.shape[1]))

	#actual predictions
	outputs = np.zeros((Y.shape[0],Y.shape[1]))

	#sum of changes we need to apply to the weights, will be 'averaged' later
	dEdWs = []
	for i in range(len(W)):
		dEdWs.append(np.zeros((W[i].shape[0],W[i].shape[1])))

	error = []

	iteration = 0
	for x,y in zip(X,Y):
		z, a = forwardProp(W, x, activationfunc)
		layers = z
		outputs = a

		#The error function
		error.append((0.5 * np.power(outputs - y,2))[0,0])
		
		#print('a', outputs)
		#print('y', Y)

		#partial derivative of output with respect to the error
		dEda = (outputs - y).T

		dEdW = backprop(dEda, W, layers, activationfunc)
		for i in range(len(W)):
			dEdWs[i] += dEdW[i]
		if iteration%(len(X)/10) == 0:
			print('*', end='', flush = True)
		iteration += 1
	print('')
	for i in range(len(W)):
		W[i] -= np.matrix(dEdWs[i]) / len(X) * learningrate


	return W, np.mean(error)

def backprop(dEda, W, layers, activationfunc, k = 1):
	if len(W) - k < 0:
		return []

	cols = W[-k].shape[1]
	rows = W[-k].shape[0]

	#partial derivative of outputs of neurons (without sigmoid) with respect to final output
	dadz = np.zeros((rows, rows))

	temp = activationfunc(layers[-k], deriv = True)
	for i in range(rows):
		dadz[i,i] = temp[0,i]
	#print('dadz\n', dadz)

	#partial derivative of neuron output (without sigmoid) with respect to the error function
	dEdz = dadz.dot(dEda)

	#partial derivative of weights with respect to error function.
	dEdw = []
	for r in dEdz:
		dEdw.append((activationfunc(layers[-k-1]) * r[0,0]).tolist()[0])
	dEdw = np.matrix(dEdw)

	#partial derivative of neuron output (without sigmoid) with respect to the actual output
	dzda = W[-k].T
	#partial derivative of neuron output in the previous layer (without sigmoid) with respect to the error function
	dEdap = dzda.dot(dEdz)

	#recursively go through all layers
	t = backprop(dEdap, W, layers, activationfunc, k + 1)
	#save changes to weights so we can caluculate the average later
	t.append(dEdw)
	return t




#preprocessing the data
_, labels = readOutput('hasy-data-labels.csv')
chars = np.matrix(labels)[:,0]
charlist = []

for i in chars:
	charlist.append(int(i[0,0]))	
X = np.matrix(readImg(charlist))
Y = []
for i in range(len(charlist)):
	Y.append(labelToExpected(labels[i], 36))
Y = np.matrix(Y)

#EDIT THIS TO CHANGE LAYERS
network = [X.shape[1], 100, 50, Y.shape[1]]

W = initNetwork(network)

#create a series of sets for mini batch training.
sets = [i for i in range(len(X))]
shuffle(sets)

#EDIT THIS TO CHANGE batch size
bs = 100
#EDIT THIS TO CHANGE amount of times to train per batch (epochs)
repeats = 100

trainingsets = int(len(X)/bs)

#info
print('Network layout', network)
print("batch size:", bs)
print("Amount of training sets:", trainingsets)
print("Repeats per training:", repeats)
print("\nStart of training:")

#the actual training
globalerror = 0
for j in range(repeats):
	batcherror = 0
	print("Repeat: ", j+1, "/", repeats)
	for i in range(trainingsets):
		print('Training set:', i+1, "/", trainingsets)
		W, error = train(W, X[sets[i*bs:i*bs+bs]], Y[sets[i*bs:i*bs+bs]],sigmoid, 1)
		batcherror += error
	batcherror /= repeats
	globalerror += batcherror
	print('Batch error:', batcherror, '\n')
globalerror /= trainingsets

print("Done!")
print("Global error:", globalerror)

#printing result
print(forwardProp(W, X[0], sigmoid)[-1].flatten(),'\n', Y[0])
for i in range (100):
	print(np.argmax(forwardProp(W, X[i], sigmoid)[-1]), np.argmax(Y[i]))

#plt.imshow(X[0])
#plt.show()

#super small test set, (AND) this one works perfectly
# x = np.matrix([[0,0], [0,1], [1,0], [1,1]])
# y = np.matrix([[1,0], [1,0], [1,0], [0,1]])
# network = [2, 10, 2]
# W = initNetwork(network)
# for i in range(100000):
	# W, error = train(W, x, y, sigmoid, 1)
	# print(error)
