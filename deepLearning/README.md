## 一个简单的神经网络

模块为文件`NN.py`, 需要了解的接口如下.

```python
class NN:
    def __init__(self,data,layers,function):
    def predict(self,parameters = None,X = None,activation = None,predictFunction = None):# it's
    def accuracy(self,predictions,labels,accuracyFunction = None):
    def train(self,learningRate,trainTimes,printCostTimes = None):
```

文件里定义了1个名为NN的类, NN代表neural network.

类中的方法作用如下(概览, 无详细说明).

| 方法名称   | 作用                       |
| ---------- | -------------------------- |
| `__init__` | 初始化神经网络             |
| `predict`  | 预测给入的数据为哪一个标签 |
| `accuracy` | 计算正确率                 |
| `train`    | 训练网络                   |

主要有2个地方是最重要的, `__init__`和`train`.

### `__init__`

用来初始化网络, 如果你要训练一个神经网络, 需要告诉它你提供的**数据**, 网络**层数**, 以及每层有多少个节点.

看这个函数的声明如下:

```python
def __init__(self,data,layers,function):
```

参数的意义如下

#### data

数据类型: dict

```python
data = {
    'trainX':yourTrainingData, # 你要训练的数据
    'trainY':yourLabels # 这些数据对应的标签
}
```

#### layers

数据类型: list

```python
layers = [TrainXSize, # 训练数据的规模
          l1Size, # 第1层网络节点数
          l2Size, # 第2层网络节点数
         	...   # 
          lnSize  # 第n层网络节点数
         ]
```

#### function

数据类型: dict

```python
function = {
	'activation':{}, # function['activation'][i]代表第i层网络的激活函数
	'derivative':{}, # function['derivative'][i]代表第i层网络的激活函数的导数
	'lostFunction':,# 损失函数
	'predictFunction': ,# 预测函数
	'accuracyFunction':# 计算正确率的函数
}
```

### `train`

```python
def train(self,learningRate,trainTimes,printCostTimes = None):
```

####learningRate

学习率

#### trainTimes

训练次数

#### printCostTimes

每隔`printCostTimes`次训练后, 打印Cost