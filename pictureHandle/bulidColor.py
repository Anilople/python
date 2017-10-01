# coding=utf-8
# 别删上边的注释，可能会编码错误
# 在python相同目录下，需要一个名字为"1.txt"的文本文件和一个名字为"test.jpg"的文件 #
# 假设txt文件中，第一行为点的个数，剩下每行的信息是 "height width r g b", 每个值都是整数
# 假设图片为一个矩形，并且每个坐标都有像素点的数据
from PIL import Image, ImageDraw, ImageFont,ImageFilter
from numpy import *
from pylab import *

fileName='out.txt' # txt文件名字在这里
pic = open(fileName)
NumberOfPoint=pic.readline() # 获取点的数量
NumberOfPoint=int(NumberOfPoint,10) # 并将其从字符转为数字
# print NumberOfPoint # 打印点的数量
l = pic.readlines() # 将剩下的每行转为字符串，放到列表里
pic.close() # 读取完文件，关闭它

l=map(lambda x:x.split(),l) # 分裂，将空格隔开的数字提取成单独的字符串
l=map(lambda x:map(lambda y:int(y,10),x),l) # 将字符串形式的数字(十进制)转为真正的数字，
# l=map(lambda x:map(lambda y:int(y,16),x),l) # 如果txt里的数字是16进制格式的，就用这一行

# 每行的信息为 "height widht r g b"
# 所以row[0]为height, row[1]为width
height=max(map(lambda row:row[0],l))+1 # 获取图片的高
width=max(map(lambda row:row[1],l))+1 # 获取图片的宽
print width,height

splitArr=[0]
for i in range(1,NumberOfPoint):
    if l[i][0]!=l[i-1][0]:
        splitArr.append(i)
print splitArr

forImArray=[]
for h in range(height):
    tempWidths=l[h*width:h*width+width] # 提取出某个高度level上的所有原始数据
    tempWidths=map(lambda x:x[2:],tempWidths)
    forImArray.append(tempWidths)
# print forImArray

imArray=array(forImArray)

# print imArray
# imArray=uint8(imArray)
# im=Image.fromarray(imArray)
# im.save('ok.jpg') # 保存图片
# im.show() # 显示图片
