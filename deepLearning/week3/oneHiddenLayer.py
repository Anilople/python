import numpy as np
import matplotlib.pyplot as plt 
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets
import pylab

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]
    n_h = 4
    h_y = Y.shape[0]
    return (n_x,n_h,h_y)

# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    params = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return params

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    Z1 = np.dot(parameters['W1'],X) + parameters['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters['W2'],A1) + parameters['b2']
    A2 = sigmoid(Z2)
    cache = {
        'Z1':Z1,
        'A1':A1,
        'Z2':Z2,
        'A2':A2   
        }
    return A2,cache

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    assert(A2.shape[0] == 1)
    assert(Y.shape[0] == 1)
    numberOfExamples = A2.shape[1]
    assert( Y.shape[1] == numberOfExamples)

    # yloga = lambda y,a:y*np.log(np.where(a < 1e-10,1e-10,a))
    # losts = yloga(Y,A2) + yloga(1-Y,1-A2)
    losts =  np.multiply(Y,np.log(A2)) + np.multiply((1 - Y),np.log(1 - A2))
    assert(losts.shape[0] == 1)
    assert(losts.shape[1] == numberOfExamples)
    cost = -1.0 / numberOfExamples * np.sum(losts) 
    assert(isinstance(cost,float))
    return cost
    
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    assert(X.shape[0] == 2)
    assert(Y.shape[0] == 1)
    numberOfExamples = X.shape[1]
    assert(Y.shape[1] == numberOfExamples)
    dZ2 = cache['A2'] - Y # (1,number of example)
    dW2 = np.dot(dZ2,cache['A1'].T) / numberOfExamples # (1,4)
    db2 = np.sum(dZ2,axis=1,keepdims=True) / numberOfExamples # (1,number of example)
    gz1 = 1 - np.power(cache['A1'],2) # (4,number of example)
    dZ1 = np.dot(parameters['W2'].T,dZ2) * gz1 # (4, number of example)
    dW1 = np.dot(dZ1,X.T) / numberOfExamples# (4,2)
    db1 = np.sum(dZ1,axis=1,keepdims=True) / numberOfExamples # (4,number of example)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

# GRADED FUNCTION: update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    W1 = parameters['W1'] - learning_rate * grads['dW1']
    W2 = parameters['W2'] - learning_rate * grads['dW2']
    b1 = parameters['b1'] - learning_rate * grads['db1']
    b2 = parameters['b2'] - learning_rate * grads['db2']
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    n_x,n_hNothing,n_y = layer_sizes(X,Y)
    parameters = initialize_parameters(n_x,n_h,n_y)
    costs = []
    for i in range(num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            print('after ' + str(i) + ' iteration cost: ' + str(cost))
        grads = backward_propagation(parameters,cache,X,Y)
        # parameters = update_parameters(parameters,grads,learning_rate=0.)
        parameters = update_parameters(parameters,grads)
        
    return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    # Z1 = np.dot(parameters['W1'],X) + parameters['b1']
    # A1 = np.tanh(Z1)
    # Z2 = np.dot(parameters['W2'],A1) + parameters['b2']
    # Y = sigmoid(Z2)
    # Y = np.where(Y > 0.5,1,Y)
    # Y = np.where(Y < 0.5,0,Y)
    A2,cache = forward_propagation(X,parameters)
    predictions = np.where(A2 > 0.5,1,A2)
    predictions = np.where(A2 < 0.5,0,predictions)
    return predictions
    

# np.random.seed(1)

# X,Y = load_planar_dataset()
# # print(X)
# print('X shape:'+str(X.shape))
# print('Y shape:'+str(Y.shape))
# print('X size:'+str(X.size))
# print('Y size:'+str(Y.size))
# print('training example:'+str(Y.size))

# X_assers, Y_assess = layer_sizes_test_case()
# n_x,n_h,n_y = layer_sizes(X_assers,Y_assess)
# print('Input layer size n_x:'+str(n_x))
# print('Hidden layer size n_x:'+str(n_h))
# print('Output layer size n_x:'+str(n_y))

# n_x, n_h, n_y = initialize_parameters_test_case()

# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# X_assess, parameters = forward_propagation_test_case()

# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


# A2, Y_assess, parameters = compute_cost_test_case()

# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# X_assess, Y_assess = nn_model_test_case()
# X_assess = 1.0/np.max(X_assess) * X_assess
# Y_assess = 1.0/np.max(Y_assess) * Y_assess

# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# parameters, X_assess = predict_test_case()

# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))

# parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
# plt.title("Decision Boundary for hidden layer size " + str(4))

# plt.scatter(X[0,:],X[1,:],c=Y[0,:],s=40,cmap=plt.cm.Spectral);
# plt.show()
# pylab.show()

# clf = sklearn.linear_model.LogisticRegressionCV() # 内置logistic 回归
# clf.fit(X.T,Y.T)

# print(clf.predict([[2,2]]))
# plot_decision_boundary(lambda x:clf.predict(x),X,Y[0,:])
# plt.title('Logistic Regression')
# plt.show()
# pylab.show()
# plot_decision_boundary

# predictions = predict(parameters, X)
# print(Y)
# print(predictions)
# print(np.abs(predictions-Y))
# print(Y.size)
# rate = 100.0 - 100.0/Y.size * np.sum(np.abs(predictions - Y))
# print(rate)
# print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations = 5000)
#     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

# plt.show()
# pylab.show()

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

print(X.shape)
print(Y.shape)

# parameters = nn_model(X,Y,5,num_iterations=10000,print_cost=True)



# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.scatter(X[0, :], X[1, :], c=Y[0,:], s=40, cmap=plt.cm.Spectral)

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0,:])
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


plt.show()
pylab.show()

