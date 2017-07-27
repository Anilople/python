# coding=utf-8
# 别删上边的注释，可能会编码错误
# 在python相同目录下，需要一个名字为"1.txt"的文本文件和一个名字为"test.jpg"的文件 #
from PIL import Image, ImageDraw, ImageFont,ImageFilter

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
# print l # 打印所有点的信息

xMax=max(map(lambda row:row[0],l)) # 最大的x坐标
yMax=max(map(lambda row:row[1],l)) # 最大的y坐标
print xMax,yMax

name='test.jpg'
im = Image.open(name) # 打开文件，这里用在原文件上修改的方式来生成图片
height=xMax 
width=yMax
im.size=height,width # 重置图片大小
# print im.size # 打印图片的尺寸

draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个

for i in range(NumberOfPoint): # 遍历所有的点, 并将其画出
    point=l[i] # 获取点的信息
    coordinate=tuple(point[0:2]) # 从点的信息中获取坐标
    color=tuple(point[2:5]) # 从点的信息中获取颜色
    draw.point(coordinate,color)

im.save('ok.jpg') # 保存图片
im.show() # 显示图片