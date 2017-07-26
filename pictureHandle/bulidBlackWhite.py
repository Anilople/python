# coding=utf-8
# 别删上边的注释，可能会编码错误
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
# # im.size 返回图片的宽和高，一个二元组
width=len(l)
height=max(map(lambda x:len(x),l))
im.size=height,width
# print size
# # im.thumbnail((480,270),Image.ANTIALIAS) # 重置图片大小
# im.thumbnail(size) # 这个也能重置图片大小，size是二元组

print im.size
draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个

for i in range(height): 
    for j in range(width): 
        coordinate=(i,j) # 要绘制的点的坐标
        # coordinate=(width-i,j) # 要绘制的点的坐标
        
        color=l[j][i]
        fill=(color,color,color) # 要绘制的点的颜色,fill(R,G,B)
        # if j<2:fill=(255,0,0)
        # if i<5 and j<5:fill=(255,255,0)
        draw.point(coordinate,fill) # 将这个点上色
        if i<2 and j<2:
            print color
            draw.point(coordinate,(255,0,0))
        # t = im.getpixel((i,j)) # 得到这个点的颜色
        # if white(t):im.putpixel((i,j),(255,255,255))


# # print color

# draw.line((0, 0) + im.size, fill=128)
# draw.line((0, im.size[1], im.size[0], 0), fill=128)

# ImageDraw.Draw.point()
# print getLine(0,0,10,10)
im.save('ok.jpg') # 保存图片
im.show()