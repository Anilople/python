import numpy as np
import random

class NN:
    def __init__(self,data,layers,function):
        """
        data -- dict
            data['trainX']
            data['trainY']
        laysers --  iterable
                    layers[0] = n_x -- size of input
                    len(layers)-1 is the number of layers, i.e. (n_h + 1) = L
                    layers[-1] = n_y -- size of output
        parameters -- python dictionary containing your parameters:                   
                        Wi -- weight matrix of shape (layers[i], layers[i-1]), LEFT MUL
                        bi -- bias vector of shape (layers[i], 1)
        function -- dict
                 -- activation: dict. Zi -> Ai
                 -- derivative: dict. dAi -> dZi
                 -- lostFunction: (AL,Y) -> cost_i
        """
        assert(isinstance(data,dict)),'data is not a dict'
        assert(isinstance(layers,list)),'layers is not a list'

        self.data = data
        self.caches = {'A0':data['trainX']} # save Zi = WiAi+bi and Ai, denote A0 = trainX
        self.Y = data['trainY']
        self.L = len(layers) - 1
        # Wi, bi
        self.parameters={}
        for i in range(1,self.L+1):
            self.parameters['W'+str(i)] = np.random.randn(layers[i],layers[i-1]) / np.sqrt(layers[i-1]) # * 0.01 # 
            self.parameters['b'+str(i)] = np.zeros((layers[i],1))  
        # function
        self.function = function
        self.activation = function['activation']
        self.derivative = function['derivative']
        self.lostFunction = function['lostFunction']
        self.grads = {} # save gradients of Wi,bi in the network
    
    def __str__(self):
        data = repr(self.data) + '\n' + '-'*20 + '\n'
        caches = repr(self.caches) + '\n' + '-'*20 + '\n'
        parameters = repr(self.parameters) + '\n' + '-'*20 + '\n'
        grads = repr(self.grads) + '\n' + '-'*20 + '\n'
        return parameters+grads+caches

    def forwardOneLayer(self,Wi,AiPrevious,bi,activation):
        Zi = np.dot(Wi,AiPrevious) + bi
        Ai = activation(Zi)
        return Zi,Ai

    def forwardPropagation(self):
        for i in range(1,self.L+1):
            Wi = self.parameters['W'+str(i)]
            AiPrevious = self.caches['A'+str(i-1)] # !! i-1 not i
            bi = self.parameters['b'+str(i)]
            self.caches['Z'+str(i)],self.caches['A'+str(i)] = self.forwardOneLayer(Wi,AiPrevious,bi,self.activation[i])

    def backwardOneLayer(self,dZi,Wi,AiPrevious,compute_dAiPrevious = True): # Zi = Wi * A_pre + bi
        # cost = sum(lost{i})/dataSize , {i} mean the i-th train data
        dataSize = dZi.shape[1] 
        dWi = np.dot(dZi,AiPrevious.T)#/dataSize
        dAiPrevious = None # according compute_dAiPrevious' value to decide whether to compute dAiPrevious
        # since AiPrevious == A0, there are meanless or cost too much resource
        if compute_dAiPrevious:dAiPrevious = np.dot(Wi.T,dZi)
        dbi = np.sum(dZi,axis=1,keepdims=True)#/dataSize # so dbi{j} = sum(dZi{j})
        assert(dWi.shape == Wi.shape)
        return dWi,dAiPrevious,dbi

    def backwardPropagation(self):
        dataSize = self.Y.shape[1]
        # compute dZL
        ZL,AL,Y = self.caches['Z'+str(self.L)],self.caches['A'+str(self.L)],self.Y
        dZL = 1/dataSize * self.derivative[self.L](ZL,AL,Y) # watch here
        self.grads['dZ'+str(self.L)] = dZL # add dZ{L} to gradient
        for i in reversed(range(1,self.L+1)):
            dZi = self.grads['dZ'+str(i)]
            Wi, AiPrevious, bi = self.parameters['W'+str(i)], self.caches['A'+str(i-1)], self.parameters['b'+str(i)]
            assert(dZi.shape[0] == bi.shape[0]),'dZi.shape[0] != bi.shape[0]'
            dWi,dAiPrevious,dbi = self.backwardOneLayer(dZi,Wi,AiPrevious,i>1) # compute gradient of Wi,bi
            # add gradient in self.grads
            self.grads['dW'+str(i)] = dWi
            self.grads['db'+str(i)] = dbi
            if i > 1 : # since there are no dA0 and dZ0
                self.grads['dA'+str(i-1)] = dAiPrevious
                # print('dA'+str(i-1),dAiPrevious)
                self.grads['dZ'+str(i-1)] = dAiPrevious * self.derivative[i-1](self.caches['Z'+str(i-1)],AiPrevious) # don't forget multiple dAiPrevious
                # print('dZ'+str(i-1),self.grads['dZ'+str(i-1)])
                
            
    def updateParameters(self,learningRate):
        for i in range(1,self.L+1):
            self.parameters["W" + str(i)] -= learningRate * self.grads['dW'+str(i)]
            self.parameters["b" + str(i)] -= learningRate * self.grads['db'+str(i)]

    def computeCost(self,AL=None,Y=None,lostFunction=None):
        """
        lostFunction : (A[L]{i},Y{i}) -> cost{i}
        """
        if AL is None:AL = self.caches['A'+str(self.L)]
        if Y  is None:Y = self.Y
        if lostFunction is None:lostFunction=self.lostFunction 
        dataSize = Y.shape[1]
        assert(AL.shape == Y.shape),"AL.shape != Y.shape"
        losts = lostFunction(AL,Y)
        cost = np.sum(losts) / dataSize
        assert(cost.shape == ())
        return cost

    def predict(self,parameters = None,X = None,activation = None,predictFunction = None):# it's not general for predict
        # compute A[L]
        A = X # input data
        # is optional parameter is None, use class's function build in
        if parameters is None: parameters = self.parameters
        if X is None: A = self.caches['A0']
        if activation is None: activation = self.activation
        if predictFunction is None: predictFunction = self.function['predictFunction']
        # compute the last layer, i.e A[L]
        for i in range(1,self.L+1):
            Wi,bi = parameters['W'+str(i)],parameters['b'+str(i)]
            Zi = np.dot(Wi,A) + bi
            A = activation[i](Zi)
        return predictFunction(A)

    def accuracy(self,predictions,labels,accuracyFunction = None):
        assert(predictions.shape == labels.shape),'prediction.shape != labels.shape'
        if accuracyFunction is None: accuracyFunction = self.function['accuracyFunction']
        return accuracyFunction(predictions,labels) # take 2 parameters

    def train(self,learningRate,trainTimes,printCostTimes = None):
        assert(type(learningRate)==float),'learningRate is not float'
        assert(type(trainTimes)==int),'train times is not int'
        costs = [] # save every printCostTimes
        for i in range(trainTimes):
            self.forwardPropagation()
            self.backwardPropagation()
            self.updateParameters(learningRate)
            if type(printCostTimes)==int and i % printCostTimes == 0:
                cost = self.computeCost()
                costs.append(cost)
                print('cost after',str(i),'iteration:',cost)
        return costs

if __name__ == 'main':
    pass
else:
    print('my Neural Network import succeed')