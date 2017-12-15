import numpy as np
import time
import math

a = np.array([1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,0])
b = np.array([0,0,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,0,1,0,0,1,1,1,0,0,0,1,1,1,0])

# a = np.arange(10)
# # b = np.squeeze(a)
# print(a)
# print(a.shape)
# print(a.dtype)
a = a.reshape(1,50)
b = b.reshape(1,50)
print(a)
print(b)
print(a.shape,b.shape)
c = b - a
print(c)
print(c.shape)
rate = 1.0 * np.sum(np.abs(c))/a.shape[1] * 100.0

print(rate)
# a = np.where(a != 0,a,1e-10)
# print(a)
# print(a.shape)
# print(a.dtype)
# b = np.log(a)
# # print(np.log(1e-10))
# print(b)
# print(b.shape)
# bsum = np.sum(b)
# print(bsum)
# print(bsum.shape)
# c = np.sum(a,axis=1)
# print(c)
# print(c.shape)
# b = np.random.rand(10,1)
# print(b.shape)

# a = np.sum(a)
# print(a)
# a=a.reshape(1,10)
# b=np.squeeze(a)
# print(b.shape)
# print(b)
# b = np.random.rand(1000000)


# tic = time.time()
# c=np.exp(a)
# toc = time.time()

# print(c)
# print("Vectorized version:"+ str(1000*(toc-tic)) + 'ms')

# tic = time.time()
# c=0
# for i in range(1000000):
#     c += a[i]*b[i]
# toc = time.time()
# print(c)
# print("For version:"+ str(1000*(toc-tic)) + 'ms')
# --------------------------------------------------------

# A = np.array([  [56.0,0.0,4.4,68.0],
#                 [1.2,104.0,52.0,8.0],
#                 [1.8,135.0,99.0,0.9]])

# print(A.shape)
# A=A.reshape(1,A.size)
# print(A)


# b=np.array([1,2,3])
# bb=np.array([[1],[2],[3]])
# print(b)
# print(bb)
# cal=b+bb
# print(cal)

# Normalizing rows
# x=np.array([[0, 3, 4],
            # [2, 6, 4]])

# x_norm = np.linalg.norm(x,ord=2,axis=1,keepdims=True)
# x_norm = np.linalg.norm(x,axis=1)
# print(x_norm)