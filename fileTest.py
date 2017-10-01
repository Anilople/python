#coding=utf-8
import os
import sys

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

scriptName=sys.argv[0] 
for i in range(1,len(sys.argv)):
    print "arguments:",i,sys.argv[i]
