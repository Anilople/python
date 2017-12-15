# coding=utf-8
import math

FPin=lambda i,n:(1+i)**n
PFin=lambda i,n:1/FPin(i,n)

FAin=lambda i,n:((1+i)**n-1)/i
AFin=lambda i,n:FAin(i,n)

APin=lambda i,n:i*(1+i)**n/((1+i)**n-1)
PAin=lambda i,n:1/APin(i,n)

getRepeat=lambda time,value:[value for i in range (time)]



def getNPV(valueArr,i):
    ans=0
    for t in range(len(valueArr)):
        ans+=FToP(valueArr[t],i,t)
    return ans

def getIRR(valueArr,ran):
    le=0.0
    ri=2.0
    while abs(le-ri)>ran:
        mid=(le+ri)/2
        # print mid
        temp=getNPV(valueArr,mid)
        if temp>0:
            le=mid
            # print '>0',mid,le,ri
        else:
            ri=mid
            # print '<0',mid,le,ri
    return le


# 假设f(x)在[left,right]上单调
def findX(f,left,right,target,pricision):
    ans=left
    while abs(f(ans)-target)>pricision: # 当精度不满足所要求的pricision时
        ran=(right-left)*1.0
        # newX=[left,(left+right)/2.0,right]
        newX=map(lambda x:left+x*0.1*ran,range(11))
        newX=sorted(newX,cmp=lambda x1,x2:cmp(abs(f(x1)-target),abs(f(x2)-target))) # 根据f(x)接近target的程度，对x进行排序
        # print "newX",newX
        fxGreaterTarget=filter(lambda x:f(x)>target,newX)   # 选取出f(x)>target的x来, 这个操作并不会改变原先x的顺序
        fxLowerTarget=filter(lambda x:f(x)<target,newX)     # 选取出f(x)<target的x来，这个操作并不会改变原先x的顺序
        # print "G",fxGreaterTarget
        # print "L",fxLowerTarget
        leftAndRight=[fxGreaterTarget[0],fxLowerTarget[0]] # 新的left和right一定在这两个当中
        leftAndRight.sort() # 排序一下
        [left,right]=leftAndRight # 获取新的left和right
        # print "Left",left
        # print "Right",right
        ans=left
    return ans

print findX(lambda x:math.exp(-x*math.pi/math.sqrt(1-x**2)),0.000001,0.99999,0.1,0.00000001)