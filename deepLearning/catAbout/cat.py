# coding=utf-8
import numpy as np 
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

import lr_utils
import pylab

sigmoid=lambda z:1.0/(1+np.exp(-z))
initialize_with_zeros=lambda dim:(np.zeros(shape=(dim,1),dtype=np.float32),0)

# w,b=initialize_with_zeros(3)

def propagate(w,b,X,Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    Z = np.dot(w.T,X)+b # (1,number of examples)
    A = sigmoid(Z) # (1,number of examples)
    # da = -1*Y/A + (1-Y)/(1-A) # (1,number of examples)
    dz = A - Y # (1,number of examples)
    # dw = np.dot(X,(A-Y).T) # (1,num_px * num_px * 3)
    dw = np.dot(X,dz.T)
    db = np.sum(dz) # scalar 

    # Aremove0 = np.where(A != 0,A,1e-6)
    # logA = np.log(Aremove0)
    # A1minusA = 1 - A
    # A1minusAremove0 = np.where(A1minusA != 0,A1minusA,1e-6) # log1A fix
    # log1A = np.log(A1minusAremove0)
    # Lost = -1 * (Y * logA + (1-Y) * log1A) # (1, number of examples)
    Lost = -1 * (Y * np.log(A) + (1-Y) * np.log(1-A))

    numberOfExamples=Y.shape[1]
    cost = np.sum(Lost)/numberOfExamples
    # cost = np.squeeze(cost)
    dw = dw/numberOfExamples
    db = db/numberOfExamples
    # db = np.squeeze(db)
    # db = np.sum(dz)/numberOfExamples
    assert(cost.shape == ())
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    # assert(db.shape == b.shape)
    return cost,dw,db

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    dw = []
    db = []
    for i in range(num_iterations):
        cost,dw,db=propagate(w,b,X,Y)
        # print cost
        # print('Cost after iteration ' + str(i) + ' ' + str(cost)) 
        if i % 100 is 0:
            costs.append(cost)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if print_cost and ( i % 100 is 0):
            print('Cost after iteration ' + str(i) + ' ' + str(cost)) 
    params={'w':w,'b':b}
    grads={'dw':dw,'db':db}
    return params,grads,costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    Z = np.dot(w.T,X) + b 
    A = sigmoid(Z)
    # print(A)
    # print(A.shape)
    Y_prediction = A
    # Y_prediction[Y_prediction<0.5] = 0
    # Y_prediction[Y_prediction>0.5] = 1
    Y_prediction=np.where(Y_prediction < 0.5, 0, Y_prediction)
    Y_prediction=np.where(Y_prediction > 0.5, 1, Y_prediction)
    # Y_prediction=np.squeeze(Y_prediction)
    assert(Y_prediction.shape == (1,X.shape[1]))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    w,b = initialize_with_zeros(X_train.shape[0])
    params,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = params['w']
    b = params['b']
    Y_prediction_test = predict(w,b,X_test)
    # print('Y_prediction_test:'+str(Y_prediction_test))
    Y_prediction_train = predict(w,b,X_train)
    # print('Y_prediction_train:'+str(Y_prediction_train))
    d = {'costs':costs,
    'w':w,
    'b':b,
    'Y_prediction_test':Y_prediction_test,
    'Y_prediction_train':Y_prediction_train,
    'learning_rate':learning_rate,
    'num_iterations':num_iterations}
    train_accurary = 100 - 100.0 * np.sum(np.abs(Y_prediction_train - Y_train))/Y_train.shape[1] 
    test_accurary = 100 - 100.0 * np.sum(np.abs(Y_prediction_test - Y_test))/Y_test.shape[1] 
    
    print("train accurary: " + str(train_accurary) + '%')
    print("test accurary: " + str(test_accurary) + '%')
    return d

# print('test')
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
orig=[train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes]
# print('train_set_y:'+str(train_set_y))
# print('test_set_y:'+str(test_set_y))
# Example of a picture
# index=3
# example = train_set_x_orig[index]
# plt.imshow(example)
# pylab.show()
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# print(map(lambda x:x.shape,orig))
# print(train_set_x_orig.shape)
# print(test_set_x_orig.shape)
# print('number of training \t examples:'+str(train_set_x_orig.shape[0]))
# print('number of test \t examples:'+str(test_set_x_orig.shape[0]))
# print('height of a training image:'+str(train_set_x_orig.shape[1]))
# print('width of a training image:'+str(train_set_x_orig.shape[2]))
# train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0],train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*3,1)
print(train_set_x_orig.shape)

train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T/255.0
# test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0],test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*3,1)
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T/255.0

print(train_set_x.shape)
# print(test_set_x.shape)
# print(len(train_set_x))
# print('sigmoid([0,2]='+str(sigmoid(np.array([0,2]))))
# w = np.array([[1],[2]])
# b = 2
# X = np.array([[1,2],[3,4]])
# Y = np.array([[1,0]])
# cost,dw,db = propagate(w,b,X,Y)
# print('cost:\t',cost)
# print('dw:\t',dw)
# print('db:\t',db)
# params, grads, costs = optimize(w,b,X,Y,num_iterations=100,learning_rate=0.009,print_cost=False)
# print('w = ' + str(params['w']))
# print('b = ' + str(params['b']))
# print('dw = ' + str(grads['dw']))
# print('db = ' + str(grads['db']))

# print("predictions = " + str(predict(w,b,X)))

d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 1500,learning_rate=0.005,print_cost=True)
# print(d)
# for i in range(50):
#     plt.imshow(test_set_x[:,i].reshape(64,64,3))
#     pylab.show()
#     myPrediction = d['Y_prediction_test'][0,i]
#     print('y = ' + str(test_set_y[0,i]) + ', you predicted that it is a ' + str(myPrediction))

# costs = np.squeeze(d['costs'])
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title('Learning rate = ' + str(d['learning_rate']))
# plt.show()
# pylab.show()

# learning_rates = [0.01,0.001,0.0001]
# models = {}
# for i in learning_rates:
#     print('Learning rate is: ' + str(i))
#     models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 1500,learning_rate=i,print_cost=False)
#     print('\n ------------------------------------------------\n')

# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]['costs']),label = str(models[str(i)]['learning_rate']))

# plt.ylabel('cost')
# plt.xlabel('iterations')

# legend = plt.legend(loc='upper center',shadow = True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()
# pylab.show()
## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "4.jpg"   # change this to the name of your image file 
## END CODE HERE ##
num_px = 64
# We preprocess the image to fit your algorithm.
# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
image = np.array(ndimage.imread(my_image, flatten=False))

my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
pylab.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")