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

y = People('Tom',12,70)
y.speak()
print(y.name)
print(y)
