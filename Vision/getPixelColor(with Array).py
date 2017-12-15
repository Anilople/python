#coding=utf-8
from PIL import Image
from pylab import *

imageName = 'ok.jpg'
outFileName = 'out.txt'
imArr = array(Image.open(imageName))
# print imArr.size
listToString=f=lambda x:reduce((lambda x,y:str(x)+' '+str(y)),x) # 定义一个list到string的函数

outStringList=[] # 使用来一次性写入文件的string list，可以大幅提高写入时间
imShape = imArr.shape 
high  = imShape[0] # 图片的高
width = imShape[1] # 图片的宽
# print width,high
outStringList.append(str(high)+' '+str(width)+'\n') # 加入高和宽的信息
# print outStringList
f = open(outFileName,'w') # 覆盖写入
for h in range(high):
    for w in range(width):
        # print imArr[i,j]
        # outStringList.append(str(i)+' '+str(j)+'\n')
        # print listToString(imArr[i][j])
        outStringList.append(str(h)+' '+str(w)+' '+listToString(imArr[h][w])+'\n') # 将每行信息加入outStringList中
        # print outStringList.count()
f.writelines(outStringList) # 一次性写入outStirngList中的内容
f.close() # 关闭文件
# print outStringList