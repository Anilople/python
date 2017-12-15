# coding=utf-8
# 别删上边的注释，可能会编码错误
from numpy import *
from pylab import *
from PIL import Image
from scipy.spatial import ConvexHull
import matplotlib as plt

isBlack=lambda x:sum(x) < 255 # 简单的二值化函数 黑色返回True，白色返回White

map2=lambda f,x1:[map(f,i) for i in x1]

def getFourDotsFromHullDots(HullDots): # 从凸包集合中获取四个点
    dotSquare = lambda dot1,dot2:(dot1[0]-dot2[0])**2+(dot1[1]-dot2[1])**2
    dots=[HullDots[0]] # 先任意获取一个点
    for i in range(3):
        HullDots=filter(lambda d:dotSquare(d,dots[i]) > 400,HullDots)
        dots.append(HullDots[0])
    return dots



imageName='whitePath.jpg'
im = Image.open(imageName)
width,high=im.size
# for h in range(high):
#     for w in range(width):
#         im.putpixel((w,h),(0,255,0))

# print im.getcolors()
# imArray=array(im)
# whitePoint=[] # 白色点的坐标集合
# for h in range(high): # y
#     for w in range(width): # x
#         if isBlack(imArray[h][w]) == False: # 如果这里是白色
#             whitePoint.append((w,h)) # 放入点集中

tupleToString=lambda (x,y):str(x)+' '+str(y)+'\n'

whitePoint=[] # 所有白点的坐标
with open('whitePoints.txt','r') as f:
    rows = f.readlines()
    rows = map(lambda x:x.split(),rows)
    for [x,y] in rows:
        whitePoint.append([int(x,10),int(y,10)])

xs = map(lambda x:x[0],whitePoint)
ys = map(lambda x:x[1],whitePoint)
plot(xs,ys,'o')

hull=ConvexHull(whitePoint)

hullDots=[]
for i in hull.vertices:
    hullDots.append(whitePoint[i])

print hullDots
print getFourDotsFromHullDots(hullDots)

# for i in hull.points:
#     print i

# show()


# hull=ConvexHull(whitePoint)
# print hull

# centerX,centerY=width/2,high/2
# allX=0
# allY=0

# for (x,y) in whitePoint:
#     allX += (x-centerX)
#     allY += (y-centerY)

# print allX,allY,numOfWhite
# centerX=centerX+allX/len(whitePoint) # 更新中心
# centerY=centerY+allY/len(whitePoint)
# print centerX,centerY


# im.show()
# print array(im)
# im.point(lambda x:x*3)
# im.show()
# Image.blend(im,imbook,0.5).show()


