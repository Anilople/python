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

fileName='top250.txt' # txt文件名字在这里
f = open(fileName)
l = f.readlines()
# f.close()
l=map(lambda x:toSQLInsert(splitByTab(x)),l)
for i in l:
    print i
# print b
# l=map(lambda x:x.split(),l)
# l=map(lambda x:map(lambda y:int(y,16),x),l)
# l=filter(lambda x:len(x)>3,l)
# print l # 输入数据
