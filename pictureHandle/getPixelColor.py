from PIL import Image, ImageDraw, ImageFont,ImageFilter
# get an image

name='test.jpg'
im = Image.open(name)


outputFile=open('out.txt','w')

outString=[]
width=im.size[0]
height=im.size[1]
for i in range(width):
    for j in range(height):
        t = im.getpixel((i,j)) # 得到这个点的R,G,B
        temp=str(i)+' '+str(j)+' '+str(t[0])+' '+str(t[1])+' '+str(t[2]) # 将坐标和颜色转为字符串
        outString.append(temp) # 然后存入列表中，备用
        # outputFile.write(temp+'\n')

outputFile.write(str(len(outString))+'\n') # 写入点的数量
map(lambda x:outputFile.write(x+'\n'),outString) # 将列表中的坐标和颜色写入文件
outputFile.close()
