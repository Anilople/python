def isPrime(a):
	for i in range(2,a):
		if a%i == 0:return False
	return True

# n=input()
n=20
num=reduce(lambda x,y:x*y,range(1,n+1))
print num
y=0
while num>0:
	y+=num%10
	num/=10
print y
s='F'
if isPrime(y):s='T'
print(str(y)+s)
# print isPrime(y)
# print isPrime(54)
