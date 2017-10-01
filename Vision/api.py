# coding=utf-8
# 以上注释为字符编码设置

from PIL import Image
from pylab import *
from numpy import *

imageName = './Vision/empire.jpg' # 图片路径和名称

# pil_im = Image.open(imageName) # 打开图片
# # pil_im.thumbnail((128,128)) # 改变图片大小（变小）
# pil_im.show() 

# pil_imL = Image.open(imageName).convert('L') # 打开图片并且转化为灰度模式
# pil_imL.show()

# read image to array
im = array(Image.open(imageName))

# # plot the image
# imshow(im)

# # some points
# x = [100,100,400,400]
# y = [200,500,200,500]

# plot(x,y,'r*')
# plot(x[:2],y[:2])
# # axis('off') # 关闭坐标显示

# show()

# create a new figure
# figure()

# don't use color
# gray()


# 彩色图片的im.shape为(高，宽，3)
print im.shape, im.dtype

# im = array(Image.open(imageName).convert('L'),'f') # 加上 'f' 将数值转换为浮点数
# 灰色图片的im.shape为(高，宽)
# print im.shape, im.dtype
print im[0,0]
print im[0,0,2]
print im.max()
show()