# coding=utf-8
# 别删上边的注释，可能会编码错误
from numpy import *
from PIL import Image

# 在python相同目录下，需要一个名字为
inFileName='out.txt' # txt输入文件名字
# 的文本文件

# 假设txt文件中，第一行为点的个数，剩下每行的信息是 "height width r g b", 每个值都是整数
# 假设图片为一个矩形，并且每个坐标都有像素点的数据

# 之后输出图片
outImageName='1.jpg' # 输出图片名字


pic = open(inFileName) # 打开文件
highAndWidth=pic.readline() # 获取高和宽 "high width"
highAndWidth=highAndWidth.split() # 以空格分割为 ["high","width"]
high = int(highAndWidth[0],10) # 提取出高
width = int(highAndWidth[1],10) # 提取出宽
# 将剩下的每行转为字符串，放到列表里. 总共有high*width个点
l = pic.readlines() 
pic.close() # 读取完文件，关闭它
l=map(lambda x:x.split(),l) # 分裂，将空格隔开的数字提取成单独的字符串 ['height','width','r','g','b']
l=map(lambda x:map(lambda y:int(y,10),x),l) # 将字符串形式的数字(十进制)转为真正的数字，[height,width,r,g,b]
l=map(lambda x:x[2:],l) # 去除前边的高度和宽度坐标

imageList=[] # 二维列表

# 将一维列表转为二维列表
for i in range(high):
    imageList.append(l[i*width:i*width+width])

imageArray = array(imageList,dtype=uint8) # 将list转为 datatype 是 unit8 的 array
# print imageArray
im = Image.fromarray(imageArray) # 从array中建立照片
# print map(len,imageArray)
im.save(outImageName) # 存储照片