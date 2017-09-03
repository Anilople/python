#coding=utf-8

# Build a .html file if it is a directory, or write it(file) in the html

# there is a way to run a python script: python [pythonScriptName] [argument1] [argument2] ...
# sys.argv[0] is [pythonScriptName], which is your python script's name
import sys
import os

# return a string: <a href="name">name</a>
def HTMLGetFileLink((href,name)):# name is a string, the name of file
    return "<a href=\""+href+"\">"+name+"</a><br>"

def HTMLGetDirLink((href,name)): # get one row link
    return "<a href=\""+href+".html\"><b>"+name+"</b></a><br>"

def HTMLInsertBody(body): # insert one row in the html body
    return '''<!doctype html>
<html>
	<head>
		<meta charset="UTF-8" />
	</head>
	<body>'''+body+'''</body>
</html>'''

def HTMLInsertBodys(bodys): # insert many rows in the html body
    return '''<!doctype html>
<html>
	<head>
		<meta charset="UTF-8" />
	</head>
	<body>
    '''+reduce(lambda x,y:x+"<br>"+'\n'+y,bodys)+'''
    </body>
</html>'''

def HTMLBuildDirectory(nowPath,dirName): # dirName为一个目录的名字
    os.chdir(nowPath) # 改变工作目录到nowPath
    dirHTMLFile=open(dirName+'.html','w') # 创建一个.html文件，与dirName同目录
    # 接下来要将目录dirName里的文件夹和文件放入这个.html文件中
    listDir=os.listdir(dirName) # 获取dirName目录下的目录名以及文件名
    # print listDir
    dirNameListDir=map(lambda x:os.path.join(dirName,x),listDir) # 将dirName目录加到每个目录以及文件的前面
    # print dirNameListDir
    subDirs=filter(os.path.isdir,dirNameListDir) # 获取dirName目录的子目录名
    files=filter(os.path.isfile,dirNameListDir) # 获取dirName目录里的文件名
    subDirsLinks=subDirs # 子目录的链接
    filesLinks=files # 文件的链接
    subDirsName=map(lambda x:os.path.split(x)[1],subDirs) # 在HTML网页中显示的子目录的名字
    filesName=map(lambda x:os.path.split(x)[1],files) # 在HTML网页中显示的dirName目录里文件的名字
    # print subDirsLinks,'\n',subDirsName
    # print filesLinks,'\n',filesName
    subDirsHTML=map(HTMLGetDirLink,zip(subDirsLinks,subDirsName)) # 给子目录创建类似<a href=""></a>的链接
    filesHTML=map(HTMLGetFileLink,zip(filesLinks,filesName)) # 给文件创建类似<a href=""></a>的链接
    # print subDirsHTML
    # print filesHTML
    dirHTMLFile.write(HTMLInsertBodys(subDirsHTML+filesHTML)) # 将刚刚给子目录和文件创建的链接写入之前创建的html文档中
    dirHTMLFile.close() 
    for i in subDirsName: # 对子目录递归进行相同的操作
        nextPath=os.path.join(nowPath,dirName)
        HTMLBuildDirectory(nextPath,i)
        

scriptName=sys.argv[0] # this python script's name
directory=sys.argv[1] # Build html directory in this directory
HTMLBuildDirectory(os.getcwd(),directory)


# dirLink=map(HTMLBuildDirLink,filter(os.path.isdir,listdir))
# fileLink=map(HTMLBuildFileLink,filter(os.path.isfile,listdir))
# out=HTMLInsertBodys(dirLink+fileLink)
# f=open('index.html','w')
# f.write(out)
# f.close()
