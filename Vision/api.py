# coding=utf-8
# 以上注释为字符编码设置

from PIL import Image
from pylab import *

imageName = './Vision/empire.jpg' # 图片路径和名称

# pil_im = Image.open(imageName) # 打开图片
# # pil_im.thumbnail((128,128)) # 改变图片大小（变小）
# pil_im.show() 

# pil_imL = Image.open(imageName).convert('L') # 打开图片并且转化为灰度模式
# pil_imL.show()

# read image to array
im = array(Image.open(imageName))

# plot the image
imshow(im)

# some points
x = [100,100,400,400]
y = [200,500,200,500]

plot(x,y,'r*')
plot(x[:2],y[:2])
# axis('off') # 关闭坐标显示

show()