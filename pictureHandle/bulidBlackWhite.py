# coding=utf-8
# 别删上边的注释，可能会编码错误
# 在python相同目录下，需要一个名字为"1.txt"的文本文件和一个名字为"test.jpg"的文件 #
from PIL import Image, ImageDraw, ImageFont,ImageFilter

fileName='1.txt' # txt文件名字在这里
pic = open(fileName)
l = pic.readlines() # 每行转为字符串，放到列表里
pic.close() # 读取完文件，关闭它
l=map(lambda x:x.split(),l) # 分裂，将空格隔开的数字提取成单独的字符串
l=map(lambda x:map(lambda y:int(y,10),x),l) # 将字符串形式的数字(十进制)转为真正的数字，
# l=map(lambda x:map(lambda y:int(y,16),x),l) # 如果txt里的数字是16进制格式的，就用这一行
print l[0]

name='test.jpg'
im = Image.open(name) # 打开文件，这里用在原文件上修改的方式来生成图片
height=len(l) # l=[[0],[0],[0]] 这里高实际上是l的长度，为3，宽是1
width=max(map(lambda x:len(x),l)) # 取最大的宽
# # im.size 返回图片的宽和高，一个二元组
im.size=width,height

print im.size
draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个

for i in range(width):
    for j in range(height): 
        coordinate=(i,j) # 要绘制的点的坐标
        color=l[j][i] # 颜色实际上是先取高的下标，然后再取宽的下标
        fill=(color,color,color) # 要绘制的点的颜色,fill(R,G,B)
        draw.point(coordinate,fill) # 将这个点上色

im.save('ok.jpg') # 保存图片
im.show()