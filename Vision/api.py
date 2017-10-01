# coding=utf-8
# 以上注释为字符编码设置

from PIL import Image

imageName = './Vision/empire.jpg' # 图片路径和名称

pil_im = Image.open(imageName) # 打开图片
pil_im.show() 

pil_imL = Image.open(imageName).convert('L') # 打开图片并且转化为灰度模式
pil_imL.show()