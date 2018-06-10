np.random.seed(1) # 改变随机数种子, 方便复现bug
data = {
'trainX':trainX,
'trainY':trainY,
'testX':testX,
'testY':testY
}

layers = [128,128,64,26] # 128代表输入数据规模, layers[i] 代表第 i 层网络的节点数

relu = lambda z:np.maximum(0,z) # Relu 激活函数
relu_deriv = lambda z,a:np.where(z > 0,1,0) # Relu 激活函数的导数
sigmoid = lambda z:1/(1+np.exp(-z)) # sigmoid 激活函数
sigmoid_deriv = lambda z,a:np.multiply(a,1-a) # sigmoid 激活函数的导数

def softmax(Z): # Z is a column vector, but we need to handle when Z is a matrix, use axis = 0
    maxNumber = np.max(Z,axis = 0,keepdims=True)
    Z -= maxNumber
    Zexp = np.exp(Z)
    return Zexp / np.sum(Zexp,axis = 0,keepdims=True)

def softmaxWithY(Z,A,Y): # softmax 的导数
    assert(A.shape == Y.shape),'A.shape is not same as Y.shape'
    return A - Y


def getFunction(layers): # 初始化function
    function = {
                'activation':{},
                'derivative':{},
                'lostFunction':lambda a,y:np.sum(np.multiply(-y,np.log(a)),axis = 0), # (AL,Y) -> Lost(AL,Y)
                'predictFunction':lambda A:A>=np.max(A,axis = 0),
                'accuracyFunction':lambda A,Y:1.0/Y.shape[1] * np.sum((np.sum(A==Y,axis = 0,keepdims=True) == 26))
                }
    L = len(layers) - 1
    for i in range(1,L):
        function['activation'][i] = relu
        function['derivative'][i] = relu_deriv
    function['activation'][L] = softmax
    function['derivative'][L] = lambda Z,A,Y:softmaxWithY(Z,A,Y)
    return function

function = getFunction(layers)
importlib.reload(NN) # 代入模块
myNN = NN.NN(data,layers,function) # 初始化网络
costs = myNN.train(learningRate=3.0,trainTimes=200,printCostTimes=1) # 开始训练
plt.plot(costs) # 绘制 costs 曲线