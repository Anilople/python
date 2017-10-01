from PIL import Image
from pylab import *
from numpy import *
import os

# print os.name
# print os.getcwd()
# print os.listdir(os.getcwd())
# os.remove('out.txt')
# os.makedirs('test')
# os.rmdir('test')
# os.rename('1.txt','2.txt')
# print os.stat('2.txt')
# im=Image.open('test.jpg')#.convert('L')
# im.show()

# out = im.resize((800,600))
# out=im.rotate(45)
# out.show()
# box=(100,100,800,400)
# region=im.crop(box)

# region=region.transpose(Image.ROTATE_90)

# im=array(Image.open('fish.jpg'))
# a=range(200)
# a=map(lambda x:range(400),a)
# arr=array(a)
# print arr
# testIm=Image.fromarray(arr)
# testIm.show()
# im2=255-im
# im3=(100.0/255)*im+100
# im4=255.0*(im/255.0)**2

# pil_im=Image.fromarray(im)
# pil_im.show()
# pil_im2=Image.fromarray(im2)
# pil_im2.show()
# pil_im3=Image.fromarray(uint8(im3))
# pil_im3.show()
# pil_im4=Image.fromarray(uint8(im4))
# pil_im4.show()
# figure()
# gray()

# hist(im.flatten(),128)
# contour(im,origin="image")
# axis('equal')
# axis('off')
# im = Image.new('RGB',(400,200),255)
# # im.size 返回图片的宽和高，一个二元组
# width=im.size[0]
# height=im.size[1]
# size=width,height
# print size
# # im.thumbnail((480,270),Image.ANTIALIAS) # 重置图片大小
# im.thumbnail(size) # 这个也能重置图片大小，size是二元组

# draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个

# for i in range(width):
#     for j in range(height):
#         draw.point((i,j),fill=(0,0,j))
        # t = im.getpixel((i,j))
        # if white(t):im.putpixel((i,j),(255,255,255))
        # color.append(t)

# im.save('ok.jpg') # 保存图片
im=Image.open('fish.jpg')
im=im.resize((192,108))
im.save('test.jpg')