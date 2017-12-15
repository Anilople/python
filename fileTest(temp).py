
fileName='asmTemp.asm'
f=open(fileName)
l=f.readlines()
f.close()
l=map(lambda x:x.split(),l)

isComment=lambda s:s[0]==';'
# row = [code,comment]
# def toRow(s):
#     ans=[]
#     c=""
#     for i in range(len(s)):
#         if(isComment(s[i])):
#             ans.append(c)
#             c=s[i][1:]
#         else:
#             c=c+s[i]+" "
#     ans.append(c)
#     return ans
unwords=lambda s:reduce(lambda x,y:x+' '+y,s) if len(s)>0 else []
unwordsBy=lambda myStr,it:reduce(lambda x,y:x+it+y,myStr) if len(myStr)>0 else []
def toRow(s):
    c=map(isComment,s)
    cIndex=0
    for i in range(len(c)):
        if c[i]==True:
            cIndex=i
    code=s[:cIndex]
    comment=s[cIndex:]
    return [unwords(code),unwordsBy(comment,', ')]

l=map(toRow,l)

# for i in l:
    # i=map(lambda x:x.replace(';','\t'),i)
    # print(toRow(i))
for i in l:
    print i

outName='temp.asm'
f=open(outName,'w')
for i in l:
    f.write(i[0]+'\t'+i[1][1:]+'\n')
f.close()
