import numpy as np

relu = lambda z:np.maximum(0,z)
relu_deriv = lambda z:np.where(z > 0,1,0)
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
    L = len(layers) - 1
    parameters={}
    for i in range(1,L+1):
        # parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) * 0.01
        # parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) / np.sqrt(2.0/layers[i-1]) # relu
        parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) / np.sqrt(layers[i-1])
        parameters['b'+str(i)] = np.zeros((layers[i],1))
    return parameters

def compute_cost(AL,Y,parameters=None,lambd=0,f=lambda AL,Y:Y * np.log(AL) + (1-Y) * np.log(1-AL)):
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
    # if f != None: # watch here
    losts = f(AL,Y)
    cost = -1.0/m * np.sum(losts)
    assert(cost.shape == ())
    
    # now we have compute cost without regularization
    # for general, whether lambd = 0 or not, we cal lambd/2/m * W
    # or check if lambd = 0,then return
    if lambd == 0: return cost
    L = len(parameters) // 2
    for i in range(1,L+1):
        Wi = parameters['W'+str(i)]
        cost += np.sum(np.power(Wi,2)) * lambd / 2 / m
    return cost

def forward_propagation(X, parameters,activation,keep_prob = 1):
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
    caches = {'A0':X,'D0':np.random.rand(X.shape[0],X.shape[1]) < 1}
    for i in range(1,L+1):
        # Wi = parameters['W'+str(i)]
        # Ai_prev = cache['A'+str(i-1)]
        # bi = parameters['b'+str(i)]
        # Zi = np.dot(Wi,Ai_prev) + bi # calculate Zi
        # cache['Z'+str(i)] = Zi # push Zi in cache
        # Ai = activation[str(i)](Zi)
        # cache['A'+str(i)] = Ai
        caches['Z'+str(i)] = np.dot(parameters['W'+str(i)],caches['A'+str(i-1)]) + parameters['b'+str(i)]
        Ai = activation[str(i)](caches['Z'+str(i)])
        Di = np.random.rand(Ai.shape[0],Ai.shape[1]) < keep_prob
        caches['D'+str(i)] = Di
        caches['A'+str(i)] = Ai * Di / keep_prob if i < L else Ai # not touch AL
        # caches['A'+str(i)] = activation[str(i)](caches['Z'+str(i)])
    return caches



def backward_propagation(Y,parameters,caches,derivative,lambd = 0,keep_prob = 1):
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
    def linear_backward(dZ,A_prev,W,b,lambd = 0,keep_prob = 1,D_prev = None): # a sub function to process one step backward
        m = dZ.shape[1]
        dA_prev = np.dot(W.T,dZ) / keep_prob
        dA_prev *= D_prev if keep_prob < 1 else 1 # dAL_prev's node must shut down as same as AL_prev
        dW = 1.0/m * (np.dot(dZ,A_prev.T) + lambd * W)
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
    # becasuse the derivative of lost will appear a = 0, then divide zero
    # so we get dZL = AL - Y, becasuse dL/dAL = - y / a + (1-y) / (1-a) and dAL/dZL = a * (1-a)
    dZL = AL - Y
    AL_prev = caches['A'+str(L-1)]
    WL = parameters['W'+str(L)]
    bL = parameters['b'+str(L)]
    dAL_prev,dWL,dbL = linear_backward(dZL,AL_prev,WL,bL,lambd,keep_prob,caches['D'+str(L-1)])
    grads = {}
    grads['dA'+str(L-1)] = dAL_prev
    grads['dW'+str(L)] = dWL
    grads['db'+str(L)] = dbL
    for i in reversed(range(1,L)):
        dAi = grads['dA'+str(i)]
        # dAi_dZi = derivative[str(i)](caches['A'+str(i)],caches['Z'+str(i)])
        dAi_dZi = derivative[str(i)](caches['Z'+str(i)])
        dZi = dAi * dAi_dZi
        Ai_prev = caches['A'+str(i-1)]
        Wi = parameters['W'+str(i)]
        bi = parameters['b'+str(i)]
        dAi_prev,dWi,dbi = linear_backward(dZi,Ai_prev,Wi,bi,lambd,keep_prob,caches['D'+str(i-1)])
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

def mini_batches_random(X,Y,mini_batch_size = 64,seed = 0):
    np.random.seed(seed)
    m = X.shape[1]
    assert(m == Y.shape[1])
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    
    k = 0
    while k*mini_batch_size < m:
        mini_batch = (shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size],shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size])
        mini_batches.append(mini_batch)
        k = k + 1
    return mini_batches
    
def nn_model(X,Y,layers,num_iterations,learning_rate,activation,derivative,print_cost=False,lambd = 0,keep_prob = 1):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (AL, number of examples)
    layers -- every layer's size, layers[i] mean the size of i_th layers
                 layers[0] mean the input size
    num_iterations -- Number of iterations in gradient descent loop
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
        caches = forward_propagation(X,parameters,activation,keep_prob)
        grads = backward_propagation(Y,parameters,caches,derivative,lambd,keep_prob)
        parameters = update_parameters(parameters,grads,learning_rate)
        cost = compute_cost(caches['A'+str(L)],Y,parameters,lambd) # here we use defult lost function
        if print_cost and i % 1000 == 0:
            print('Cost after iteration '+str(i)+' '+str(cost))
        costs.append(cost)
    return parameters,costs

def nn_model_momentum(X,Y,layers,num_iterations,learning_rate,activation,derivative,print_cost=False,lambd = 0,keep_prob = 1,beta = 0.9):
    def getV(parameters):
        L = len(parameters) // 2
        v = {}
        for i in range(1,L+1):
            dWi = parameters['W'+str(i)]
            v['dW'+str(i)] = np.zeros(dWi.shape)
            v['db'+str(i)] = np.zeros((dWi.shape[0],1))
        return v
        
    def update_V(v,grads,beta = 0.9):
        L = len(v) // 2
        for i in range(1,L+1):
            dWi = grads['dW'+str(i)]
            v['dW'+str(i)] = beta * v['dW'+str(i)] + (1-beta) * grads['dW'+str(i)]
            v['db'+str(i)] = beta * v['db'+str(i)] + (1-beta) * grads['db'+str(i)]
        return v
        
    parameters = initialize_parameters(layers)
    L = len(parameters) // 2
    v = getV(parameters)
    costs = []
    for i in range(num_iterations):
        caches = forward_propagation(X,parameters,activation,keep_prob)
        grads = backward_propagation(Y,parameters,caches,derivative,lambd,keep_prob)
        v = update_V(v,grads,beta = 0.9)
        parameters = update_parameters(parameters,v,learning_rate)
        cost = compute_cost(caches['A'+str(L)],Y,parameters,lambd) # here we use defult lost function
        if print_cost and i % 1000 == 0:
            print('Cost after iteration '+str(i)+' '+str(cost))
        costs.append(cost)
    return parameters,costs

    
def nn_model_adam(X,Y,layers,num_iterations,learning_rate,activation,derivative,print_cost=False,lambd = 0,keep_prob = 1,beta1 = 0.9, beta2=0.999, epsilon = 1e-8):
    def initialize_adam(parameters):
        v = {}
        s = {}
        L = len(parameters) // 2
        for i in range(1,L+1):
            Wi = parameters['W'+str(i)]
            bi = parameters['b'+str(i)]
            v["dW" + str(i)] = np.zeros(Wi.shape)
            v["db" + str(i)] = np.zeros(bi.shape)
            s["dW" + str(i)] = np.zeros(Wi.shape)
            s["db" + str(i)] = np.zeros(bi.shape)
        return v,s
    
    def update_v_s(v,s,grads,t,beta1 = 0.9, beta2=0.999, epsilon = 1e-8):
        L = len(parameters) // 2
        for i in range(1,L+1):
            v["dW" + str(i)] = beta1 * v['dW' + str(i)] + (1-beta1) * grads['dW' + str(i)]
            v["db" + str(i)] = beta1 * v['db' + str(i)] + (1-beta1) * grads['db' + str(i)]
            vW_corrected = v["dW" + str(i)] / (1-np.power(beta1,t))
            vb_corrected = v["db" + str(i)] / (1-np.power(beta1,t))
            s["dW" + str(i)] = beta2 * s['dW' + str(i)] + (1-beta2) * np.power(grads['dW' + str(i)], 2)
            s["db" + str(i)] = beta2 * s['db' + str(i)] + (1-beta2) * np.power(grads['db' + str(i)], 2)   
            sW_corrected = s["dW" + str(i)] / (1-np.power(beta2,t))
            sb_corrected = s["db" + str(i)] / (1-np.power(beta2,t))
            grads["dW" + str(i)] = vW_corrected / (epsilon + np.sqrt(sW_corrected))
            grads["db" + str(i)] = vb_corrected / (epsilon + np.sqrt(sb_corrected))
        return v,s,grads
    parameters = initialize_parameters(layers)
    L = len(parameters) // 2
    v,s = initialize_adam(parameters)
    costs = []
    for i in range(num_iterations):
        caches = forward_propagation(X,parameters,activation,keep_prob)
        grads = backward_propagation(Y,parameters,caches,derivative,lambd,keep_prob)
        v,s,grads = update_v_s(v,s,grads,i+1,beta1 = 0.9, beta2=0.999, epsilon = 1e-8)
        parameters = update_parameters(parameters,grads,learning_rate)
        cost = compute_cost(caches['A'+str(L)],Y,parameters,lambd) # here we use defult lost function
        if print_cost and i % 1000 == 0:
            print('Cost after iteration '+str(i)+' '+str(cost))
        costs.append(cost)
    return parameters,costs
            
def predict(X, parameters,activation,predict_function = lambda AL:(AL > 0.5).astype(float)):
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

def getFuncFrom(layers):
    L = len(layers) - 1
    activation = {str(L):sigmoid}
    derivative = {str(L):sigmoid_deriv,'lost':lost}
    for i in range(1,L):
        activation[str(i)] = relu
        derivative[str(i)] = relu_deriv
    return activation,derivative

def calcuAccuracy(predictions,reals):
    assert(predictions.shape == reals.shape)
    predictions.astype(float)
    reals.astype(float)
    m = reals.shape[1]
    return 1 - 1.0/m * np.sum(np.abs(predictions - reals))