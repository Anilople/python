#coding=utf-8
from PIL import Image, ImageDraw, ImageFont,ImageFilter
from numpy import *
# 这个文件的作用是，获取某张图片中的每个像素点的信息
# 输出的格式是
# 第一行为图片的像素点个数n
# 剩余的n行，每行的数据为 "height width r g b" 
name='test.png' # 图片的名字
im = Image.open(name) # 打开图片

outputFile=open('out.txt','w') # 这里是输出文件的名字，如果文件存在，则会被覆盖

imArray=array(im) # 将图片转为array数据存储
# pirnt imArray.size

toColorString=lambda z:reduce(lambda x,y:str(x)+" "+str(y),z) # 定义一个函数，作用为 类似[1,2,3] -> "1 2 3" 
words=lambda z:reduce(lambda x,y:x+','+y,z) if len(z) > 0 else ''
# outputFile.write(str(imArray.size/3)+'\n') # 写入图片的像素点数量

widths=map(lambda x:len(x),imArray) # 获取每个高度上的宽度
width=max(widths) # 获取图片最大的宽度
height=len(imArray) # 获取图片的高度

outLines=[]
# 每行写入 "height width r g b"
# for h in range(height):
    # for w in range(width):
		# outLines.append(str(h)+" "+str(w)+" "+toColorString(imArray[h][w])+"\n")
        # if h==107 and w>81:
            # print str(h),str(w),toColorString(imArray[h][w])
        # outputFile.write(str(h)+" "+str(w)+" ") # 写入 "height width "
        # outputFile.write(toColorString(imArray[h][w])+"\n") # 写入 "r g b"
print height,width
tempRow=[]
for h in range(height):
	tempRow=[]
	for w in range(width):
		tempRow.append(1 if sum(imArray[h][w]) > 255 else 0)
	a=tempRow[0:40]
	b=tempRow[40:-1]
	a=map(str,a)
	b=map(str,b)
	a=words(a)
	b=words(b)
	outLines.append(a)
	outLines.append(b)
# print outLines
outLines=map(lambda x:'\tdb '+x+'\n',outLines)
outputFile.writelines(outLines)
outputFile.close()
# for i in range(20625,len(outLines)):
    # print outLines[i]
# print imArray[107][84]
# a=imArray[107][90]
# print toColorString(a)