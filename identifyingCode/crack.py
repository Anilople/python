# coding=utf-8
# 不要删掉上边一行的注释

from PIL import Image

def split(it,right,wrong):# 获取单片right的范围，以[(start1,end1),(start2,end2)]的形式保存
    start=[]
    end=[]
    # 找出开头
    if it[0]==right:
        start.append(0)
    for i in range(1,len(it)):
        if it[i-1]==wrong and it[i]==right:
            start.append(i)

    # 找出结尾
    for i in range(len(it)-1):
        if it[i]==right and it[i+1]==wrong:
            end.append(i)
    if it[len(it)-1]==right:
        end.append(len(it)-1)
    
    return zip(start,end)

def getColorDict(histogram):# 将直方图数据存入字典中，方便排序查询
    colorDict={}
    for i in range(len(histogram)):
        colorDict[i]=histogram[i]
    return colorDict

def getLetters(im):# 对字符进行纵向分割，提取出单个字符的范围(假设颜色只有0和255)
    row=[] # 将黑点沿着高度方向进行投影，得到一行数据。如果某列中没有黑点，那么数据为False，否则为True
    for y in range(im.size[0]):# 对宽度进行遍历
        blackInColumn=False
        for x in range(im.size[1]): #对高度进行遍历
            if im.getpixel((y,x)) != 255: # 如果这一列里有非白色的点
                blackInColumn=True
        row.append(True if blackInColumn else False) # 就在row里放入True，否则放入False [黑色为True,白色为False]
    return split(row,True,False)

# 图片路径
identifyPath='C:\Users\lambda\Desktop\python\identifyingCode\python_captcha\captcha.gif'
im=Image.open(identifyPath) # 打开图片
im.convert('P') # 将图片转为8位像素模式
# print im.histogram() # 打印颜色直方图
# 颜色直方图的每一位数字都代表了在图片中含有对应位的颜色的像素的数量

im2 = Image.new('P',im.size,255) # 创建一张新图片
for y in range(im.size[0]):# 遍历宽度
    for x in range(im.size[1]):# 遍历高度
        pixel=im.getpixel((y,x))
        if pixel == 220 or pixel == 227:
            im2.putpixel((y,x),0)

im2.show()
letters=getLetters(im2)

for charRange in letters:
    start=charRange[0]
    end=charRange[1]
    charWidth=end-start+1
    tempIm=Image.new('P',(charWidth,im2.size[1]),255)
    for width in range(charWidth):
        for height in range(im2.size[1]):
            pixel=im2.getpixel((start+width,height))
            tempIm.putpixel((width,height),pixel)
    tempIm.show()

# colorDict=getColorDict(im.histogram()) # 将historgram导入dict数据结构中
# colorDict.items() 获取[(key,value)]
# colorDictSorted=sorted(colorDict.items(),key=lambda x:x[1],reverse=True)

# im.show() # 显示图片