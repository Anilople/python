class People:
    def __init__(self,name='',age=0,weight=0):
        self.name=name
        self.age=age
        self.weight=weight
    def __del__(self):
        print('bye!','I am',self.name)
    def __repr__(self):
        return self.name+'-'+str(self.age)+'-'+str(self.weight)
    def speak(self):
        print(self.name,'说他',self.age,'岁,',self.weight,'斤')

# y = People('Tom',12,70)
# y.speak()
# print(y.name)
# print(y)

class Fib:
    def __init__(self):
        self.a = 0
        self.b = 1
    def __str__(self):
        return str(self.a)
    def __iter__(self):
        return self
    def __next__(self):
        self.a, self.b = self.b, self.a+self.b
        if self.a > 20:
            raise StopIteration() 
        return self.a
    def __getitem__(self,n):
        if isinstance(n,int):
            a, b = 0, 1
            for x in range(n):
                a, b = b, a+b
            return a
        if isinstance(n,slice):
            start = n.start
            stop = n.stop
            if start is None:
                start = 0
            a, b = 0, 1
            L = []
            for i in range(stop):
                if i >= start:
                    L.append(a)
                a, b = b, a+b
            return L

aFib = Fib()
