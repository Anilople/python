import threading
import time

def printTime(threadName,delay,delayCount = 10):
    while delayCount > 0:
        time.sleep(delay)
        delayCount -= 1
        print(threadName,time.ctime(time.time()),'---','delayCount:',str(delayCount))


class MyThread(threading.Thread):
    def __init__(self,threadID,name,delay,counter):
        threading.Thread.__init__(self)
        self.threadID =  threadID
        self.delay = delay
        self.name = name
        self.counter = counter
    def run(self):
        print("开始: "+self.name)
        printTime(self.name,self.delay,self.counter)
        print("退出: "+self.name)

thread1 = MyThread(1,'T-1',1,10)
thread2 = MyThread(2,'T-2',3,3)

thread1.start()
thread2.start()
thread1.join()
thread2.join()
print('quit')