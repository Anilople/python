#coding=utf-8

def splitByTab(it):
    ans=[]
    tempStr=""
    for i in it:
        if i != '\t':
            tempStr=tempStr+i
        else:
            ans.append(tempStr)
            tempStr=""
    ans.append(tempStr)
    return ans

def toSQLInsert(t):
    ans="INSERT INTO imdbtop250 VALUES("
    t[1]='\''+t[1]+'\''
    for i in range(len(t)-1):
        ans=ans+t[i]+','
    t[len(t)-1]=t[len(t)-1].split()[0]
    ans=ans+t[len(t)-1]+');'
    return ans

def addSQM(it):
    temp=""
    for i in it:
        if i == '\'':
            temp=temp+'\\'
        temp=temp+i
    return temp

def buildMusicHTML(musicName):
    return '''<!doctype html>
<html>
	<head>
		<meta charset="UTF-8" />
	</head>
	<body>
		<embed width="1000" src="'''+musicName+'''"/>
	</body>
</html>'''

def buildMusicLink(musicName):
    return '<a href=\"music/'+musicName+'\"'+' target=\"_blank\"'+'>'+musicName+'</a>'

fileName='1.txt' # txt文件名字在这里
f = open(fileName)
l = f.readlines()
f.close()
path="C:/Users/lambda/Desktop/temp/Web/music/"
l=map(lambda x:x.strip('\n'),l)
f=open(path+'out.txt','w')
for i in l:
    f.write(buildMusicLink(i)+'\n')
    f.write("<br></br>"+'\n')
    # f=open(path+i+".html",'w')
    # f.write(buildMusicHTML(i))
f.close()
# print b
# l=map(lambda x:x.split(),l)
# l=map(lambda x:map(lambda y:int(y,16),x),l)
# l=filter(lambda x:len(x)>3,l)
# print l # 输入数据
