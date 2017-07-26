from PIL import Image, ImageDraw, ImageFont,ImageFilter
# get an image

# def deleteStrip(cr):
#     return cr.strip()

# def deleteBeginAndLast(it):
#     it.pop()
#     return it

# fileName='2.txt' # txt文件名字在这里
# pic = open(fileName)
# l = pic.readlines()
# pic.close()
# l=map(deleteStrip,l)
# l=map(lambda x:x.split(),l)
# l=map(deleteBeginAndLast,l)
# l=map(lambda x:map(lambda y:int(y,16),x),l)
# l=filter(lambda x:len(x)>3,l)
# print l # 输入数据

name='work.jpg' # 要打开来处理的图片
im = Image.open(name)
# # im.size 返回图片的宽和高，一个二元组
# width=len(l[0])
# height=len(l)
width=256
height=256
size=width,height
print size 
# im.thumbnail(size) # 这个也能重置图片大小，size是二元组
# im.thumbnail((width,height),Image.ANTIALIAS) # 重置图片大小
im.size=size

draw = ImageDraw.Draw(im) # 在使用draw.point() 之前，先用这个
column=range(width)
# column.reverse()
row=range(height)

x=0
y=0
for i in row:
    for j in column:
        # for k in range(256)
        # black=
        # if black > 180:black=255
        # if black < 40:black=0
        # draw.point((j,i),fill=(black,black,black))
        draw.point((i,j),fill=(i,j,0))


# im.save('work.jpg') # 保存图片
im.show() # 显示图片