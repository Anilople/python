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

name='test.jpg'
im = Image.open(name)
# # im.size 返回图片的宽和高，一个二元组
width=1000
height=500
size=width,height
# print size
# # im.thumbnail((480,270),Image.ANTIALIAS) # 重置图片大小
im.thumbnail(size) # 这个也能重置图片大小，size是二元组

draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个

for i in range(width):
    for j in range(height):
        draw.point((i,j),fill=(i,j%255,i%255))
        # t = im.getpixel((i,j))
        # if white(t):im.putpixel((i,j),(255,255,255))
        # color.append(t)

im.save('ok.jpg') # 保存图片
im.show()