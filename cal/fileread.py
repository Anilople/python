
name='cal/matrix.txt'
f=open(name)
mat=f.readlines()
f.close()
mat=map(lambda x:x.split(),mat)
mat.pop()
x=mat[0]
for row in range(1,len(mat)):
    for column in range(len(mat[row])):
        print str(mat[row][column])+'*'+str(x[column])+'+',
    print ''
# print mat