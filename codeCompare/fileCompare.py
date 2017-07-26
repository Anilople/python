
def getFileLines(name):
    with open(name) as f:
        return f.readlines()

def deleteStrip(cr):
    return cr.strip()

def deleteNote(it):
    ans=''
    #status=0
    for i in range(len(it)):
        if it[i] is '/':
            return ans
        else:ans=ans+it[i]
    return ans

def getAllRow(name):
    s=getFileLines(name)
    s=map(deleteStrip,s)
    s=map(deleteNote,s)
    s=map(deleteStrip,s)
    s=filter(lambda x:len(x)>5,s)
    return s

def writeThem(name,charList):
    with open(name,'w') as f:
        for i in charList:
            f.write(i+'\n')

name="right.c"
f=open(name,'r')
print f.readlines()
f.close()

# right=getAllRow('right.c')#the right code's name
# writeThem('handleRight.c',right)
# # for i in right:
#     # print i
# # print '-' * 100
# wrong=getAllRow('wrong.c')#the wrong code's name
# writeThem('handleWrong.c',wrong)
# # for i in wrong:
# #     print i
# for i in wrong:
#     if not (right.count(i) is wrong.count(i)):
#         print i
    # found=False
    # for j in right:
    #     if i is j:found=True
    # if(not found):print i

