import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *
import pylab

relu = lambda z:np.maximum(0,z)
relu_deriv = lambda z:np.where(z > 0,1,0.01)
sigmoid = lambda z:1/(1+np.exp(-z))
sigmoid_deriv = lambda z:((lambda a:np.multiply(a,1-a))(sigmoid(z)))
lost = lambda a,y: - y / a + (1-y) / (1-a)

def initialize_parameters(layers):
    """
    Argument:
    
    laysers --  iteratable
                layers[0] = n_x -- size of input
                len(layers)-1 is the number of layers, i.e. (n_h + 1) = L
                layers[-1] = n_y -- size of output

    Returns:
    parameters -- python dictionary containing your parameters:
                    L -- number of layers                    
                    Wi -- weight matrix of shape (layers[i], layers[i-1])
                    bi -- bias vector of shape (layers[i], 1)
    """
    m = len(layers) - 1
    parameters={}
    for i in range(1,m+1):
        # parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) * 0.01
        parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) / np.sqrt(layers[i-1])
        parameters['b'+str(i)] = np.zeros((layers[i],1))
    return parameters

def compute_cost(AL,Y,f=None):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    Y.astype(float)
    m = AL.shape[1]
    assert(Y.shape[1] == m)
    if f != None: # watch here
        losts = f(AL,Y)
    losts = Y * np.log(AL) + (1-Y) * np.log(1-AL)
    cost = -1.0/m * np.sum(losts)
    assert(cost.shape == ())
    return cost

def forward_propagation(X, parameters,activation):
    """
    Argument:
    X -- input data , shape is (n_x,number of example)
    parameters -- python dictionary containing your parameters:
                L -- number of layers
                Wi -- weight matrix of shape (layers[i], layers[i-1])
                bi -- bias vector of shape (layers[i], 1)
    activation -- python dictionary containing activation function:
                activation['i'] -- the ith layer's activation function 
    Returns:
    caches -- python dictionary containing your parameters:
                Ai -- activation of ith layer
                Zi -- Z of ith layer
    """
    # L = parameters['L']
    L = len(parameters) // 2
    caches = {'A0':X}
    for i in range(1,L+1):
        # Wi = parameters['W'+str(i)]
        # Ai_prev = cache['A'+str(i-1)]
        # bi = parameters['b'+str(i)]
        # Zi = np.dot(Wi,Ai_prev) + bi # calculate Zi
        # cache['Z'+str(i)] = Zi # push Zi in cache
        # Ai = activation[str(i)](Zi)
        # cache['A'+str(i)] = Ai
        caches['Z'+str(i)] = np.dot(parameters['W'+str(i)],caches['A'+str(i-1)]) + parameters['b'+str(i)]
        caches['A'+str(i)] = activation[str(i)](caches['Z'+str(i)])
    return caches

def backward_propagatin(Y,parameters,caches,derivative):
    '''
    Arguments:
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    parameters -- A dictionary with
            Wi , bi
    caches -- A dictionary with
            Zi -- use ['Z'+str(i)] to get it  
            Ai -- use ['A'+str(i)] to get it
            they both have same size
    derivative -- A dictionary with
            (da/dz)_i,the derivative of i_th layer's activation function
            use [str(i)] to get it, it take two parameters (Zi),
                                    and return dAi/dZi, 
                                    so we can get dZi through dZi = dA(i) * (dAi/dZi) 
            use ['lost'] to get the derivative of lost function, two parameters (AL,Y) 
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    '''
    def linear_backward(dZ,A_prev,W,b): # a sub function to process one step backward
        m = dZ.shape[1]
        dA_prev = np.dot(W.T,dZ)
        dW = 1.0/m * np.dot(dZ,A_prev.T)
        db = 1.0/m * np.sum(dZ,axis=1,keepdims=True)
        assert(dW.shape == W.shape)
        assert(db.shape == b.shape)
        return dA_prev,dW,db

    Y.astype(float) # for calculate,we set boolean to float
    # L = len(caches) // 2
    # L = parameters['L'] # number of layers
    L = len(parameters) // 2
    derivative_lost = derivative['lost'] # derivative of lost function
    AL = caches['A'+str(L)] # activation of layer L, i.e. final layer
    grads = {'dA'+str(L):derivative_lost(AL,Y)} # dAL, i.e. dC/dAL
    for i in reversed(range(1,L+1)):
        dAi = grads['dA'+str(i)]
        # dAi_dZi = derivative[str(i)](caches['A'+str(i)],caches['Z'+str(i)])
        dAi_dZi = derivative[str(i)](caches['Z'+str(i)])
        dZi = dAi * dAi_dZi
        Ai_prev = caches['A'+str(i-1)]
        Wi = parameters['W'+str(i)]
        bi = parameters['b'+str(i)]
        dAi_prev,dWi,dbi = linear_backward(dZi,Ai_prev,Wi,bi)
        grads['dA'+str(i-1)] = dAi_prev
        grads['dW'+str(i)] = dWi
        grads['db'+str(i)] = dbi
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters
        Wi,bi
    grads -- python dictionary containing your gradients, output of L_model_backward
        dWi,dbi
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    # L = parameters['L']
    L = len(parameters) // 2
    for i in range(1,L+1):
        parameters["W" + str(i)] -= learning_rate * grads['dW'+str(i)]
        parameters["b" + str(i)] -= learning_rate * grads['db'+str(i)]
    return parameters

def nn_model(X,Y,num_iterations,learning_rate,layers,activation,derivative,print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (AL, number of examples)
    num_iterations -- Number of iterations in gradient descent loop
    layers -- every layer's size, layers[i] mean the size of i_th layers
                 layers[0] mean the input size
    activation -- a python dictionary
                activation['i'] -- the activation function of i_th layer
    derivative -- a python dictionary
                derivative['i'] -- the derivative of (the activation function of i_th layer)
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs -- the cost after every learning
    """
    parameters = initialize_parameters(layers)
    # L = parameters['L']
    L = len(parameters) // 2
    costs = []
    for i in range(num_iterations):
        caches = forward_propagation(X,parameters,activation)
        grads = backward_propagatin(Y,parameters,caches,derivative)
        parameters = update_parameters(parameters,grads,learning_rate)
        cost = compute_cost(caches['A'+str(L)],Y) # here we use defult lost function
        if print_cost and i % 100 == 0:
            print('Cost after iteration '+str(i)+' '+str(cost))
        costs.append(cost)
    return parameters,costs

def predict(X, parameters,activation,predict_function):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters 
    activation -- 
    predict_function -- lambda AL:(AL > 0.5).astype
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    caches = forward_propagation(X,parameters,activation)
    # L = parameters['L']
    L = len(parameters) // 2
    AL = caches['A'+str(L)]
    predictions = predict_function(AL)
    return predictions

def getFunc(layers):
    L = len(layers) - 1
    activation = {str(L):sigmoid}
    derivative = {str(L):sigmoid_deriv,'lost':lost}
    for i in range(1,L):
        activation[str(i)] = relu
        derivative[str(i)] = relu_deriv
    return activation,derivative

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))

# layers = [12288,277,191,131,97,71,43,13,7,5,1]
# layers = [12288,200,150,100,50,30,20,10,5,1]
layers = [12288,100,20,7,1]

activation,derivative = getFunc(layers)
# activation = {
#     '1':relu,
#     '2':relu,
#     '3':relu,
#     '4':sigmoid
# }
# derivative = {
#     '1':relu_deriv,
#     '2':relu_deriv,
#     '3':relu_deriv,
#     '4':sigmoid_deriv,
#     'lost':lost
# }
print(len(layers)-1)
np.random.seed(1)
parameters,costs = nn_model(train_x,train_y,4000,0.0072,layers,activation,derivative,True)

predictions = predict(train_x,parameters,activation,lambda AL:(AL > 0.5).astype(float))
predictions_test = predict(test_x, parameters,activation, lambda AL:(AL > 0.5).astype(float))
calPredict = lambda predictions,real_Y:100.0 - 100.0 * np.sum(np.abs(predictions - real_Y))/real_Y.size
print(calPredict(predictions,train_y))
print(calPredict(predictions_test,test_y))

plt.plot(costs)
plt.show()
pylab.show()

# print_mislabeled_images(classes, test_x, test_y, pred_test)
# predictions = predict(train_x,parameters,activation,lambda x:(x > 0.5).astype(float))
# print(predictions)
# print(calPredict(predictions,train_y))

# predictions = predict(test_x,parameters,activation,lambda AL:(AL > 0.5).astype(float))
# print(calPredict(predictions,test_y))

# plt.plot(costs)
# plt.show()
# pylab.show()

# activ_relu = lambda z:np.where(z > 0,z,0)
# deriv_relu = lambda a,z:np.where(z > 0,1,0)
# activ_sigmoid = lambda z:1/(1+np.exp(-z))
# deriv_sigmoid = lambda a,z:np.multiply(a,(1-a))
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]
    activ,deriv = getInitial(layers_dims)
    parameters = initialize_parameters(layers_dims)
    
    for i in range(num_iterations):
        caches = forward_propagation(X,parameters,activ)
        cost = compute_cost(caches['A2'],Y)
        grads = backward_propagatin(Y,parameters,caches,deriv)
        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

# n_x = 12288     # num_px * num_px * 3
# n_h = 7
# n_y = 1
# layers_dims = (n_x, n_h, n_y)
# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)

