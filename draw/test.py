# coding=utf-8
import matplotlib.pyplot as plt
# import numpy as np
import random
# from scipy import interpolate

# def drawAndSave(index,myRange):
#     # myRange
#     # f=open('randomPicture/'+str(index)+'.txt','w')
#     # f.writelines('myRange:\t'+str(myRange)+'\n')

#     xRandom=[]
#     yRandom=[]
#     xOrigin=0
#     yOrigin=0
#     for i in range(10000):
#         # f.writelines(str(xOrigin)+'\t'+str(yOrigin)+'\n')
#         xRandom.append(xOrigin)
#         yRandom.append(yOrigin)
#         xOrigin+=random.randint(-myRange,myRange)
#         yOrigin+=random.randint(-myRange,myRange)
#     plt.plot(xRandom,yRandom,'r')
#     # f.close()
#     plt.savefig(str(index)+'.jpg') # 保存图片
# 	# plt.show()
#     plt.clf()


a=1
b=1
def f(w):
    return 1/(a+w)/(b+w)

be=1
en=1000
ww=[]
while be<en:
    if be != (-1*a) and be != (-1*b):
        ww.append(be)
    be+=1
print ww
ww=map(lambda x:x*0.01,ww)
xw=map(f,ww)

plt.plot(xw)
plt.show()
# plt.ylabel('this is y') # y轴标签
# plt.xlabel('this is x') # x轴标签

# plt.axis([0,6,0,20]) # 只看在x轴为0-6，y轴为0-20的区域

# plt.show()