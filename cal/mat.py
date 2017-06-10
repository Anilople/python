
# import numpy
# # mat3 = numpy.diag((1,2,3))
# # print mat3

# mat=0
# name='a.txt'
# with open(name) as f:
#     mat=f.readlines()

# mat=map(lambda x:x.split(),mat)
# mat=map(lambda x:map(int,x),mat)

# mat4 = numpy.mat(mat)

# print mat4
# lamb,colu=numpy.linalg.eig(mat4)

import pulp
model = pulp.LpProblem("profit",pulp.LpMaximize)
x1=pulp.LpVariable('x1',lowBound=0)
x2=pulp.LpVariable('x2',lowBound=0)
x3=pulp.LpVariable('x3',lowBound=0)
x4=pulp.LpVariable('x4',lowBound=0)
x5=pulp.LpVariable('x5',lowBound=0)
x6=pulp.LpVariable('x6',lowBound=0)
x7=pulp.LpVariable('x7',lowBound=0)
x8=pulp.LpVariable('x8',lowBound=0)
x9=pulp.LpVariable('x9',lowBound=0)


model += 10.2*x1+ 1.8*x2+ 29*x3+ 2.4*x4+ 1.8*x5+ 4*x6+ 2.4*x7+ 8*x8+ 3.9*x9,'max'

# Constraints
model += 12*x1+ 4.8*x3+ 24*x4+ 8*x5+ 4.1*x6+ 2.4*x7+2*x9<=1690
model += 16*x1+ 1.8*x2+ 4*x4+ 1.7*x7+ 28*x8+ 9.2*x9<=986
model += 9*x3+ 7*x4+ 12*x5+ 4*x6+ 6*x7+ 8*x8+ 5*x9 <=950
model += 15*x1+ 1.4*x2+ 1*x3+ 8*x4+ 21*x5+ 1*x6+ 16*x7+ 6*x8+ 15*x9<=2017

model.solve()
print pulp.LpStatus[model.status]
# Out[5]:

print("x1 = {}".format(x1.varValue))
print("x2 = {}".format(x2.varValue))
print("x3 = {}".format(x3.varValue))
print("x4 = {}".format(x4.varValue))
print("x5 = {}".format(x5.varValue))
print("x6 = {}".format(x6.varValue))
print("x7 = {}".format(x7.varValue))
print("x8 = {}".format(x8.varValue))
print("x9 = {}".format(x9.varValue))

print(pulp.value(model.objective))

