from PIL import Image, ImageDraw, ImageFont,ImageFilter
# get an image
def getLine(x0,y0,x1,y1):
    ans=[]
    while (x0<x1 and y0<y1):
        # ans.append((x0,y0))
        ans.append(x0)
        ans.append(y0)
        x0=x0+1
        y0=y0+1
    return ans

def deleteStrip(cr):
    return cr.strip()

def deleteBeginAndLast(it):
    it.pop()
    out=[]
    return it

fileName='1.txt' # txt文件名字在这里
pic = open(fileName)
l = pic.readlines()
pic.close()
l=map(deleteStrip,l)
l=map(lambda x:x.split(),l)
l=map(deleteBeginAndLast,l)
l=map(lambda x:map(lambda y:int(y,16),x),l)
l=filter(lambda x:len(x)>3,l)
print l
# one = l[0]
# num=one.split()
# print num
# num=map(lambda x:int(x,16),num)
# print num

name='test.jpg'
im = Image.open(name)
# # im.size 返回图片的宽和高，一个二元组
width=len(l[0])
height=len(l)
size=width,height
print size
# # im.thumbnail((480,270),Image.ANTIALIAS) # 重置图片大小
im.thumbnail(size) # 这个也能重置图片大小，size是二元组

# def white(color):
#     return len(filter(lambda x:x<220,color))<1

# color=[]
# for i in range(width):
#     for j in range(height):
#         # draw.point((i,j),fill=(i%255,j%255,i%255))
#         t = im.getpixel((i,j))
#         if white(t):im.putpixel((i,j),(255,255,255))
#         color.append(t)

# # print color

draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个
# draw.line((0, 0) + im.size, fill=128)
# draw.line((0, im.size[1], im.size[0], 0), fill=128)

# ImageDraw.Draw.point()
# print getLine(0,0,10,10)
im.save('ok.jpg') # 保存图片
im.show()