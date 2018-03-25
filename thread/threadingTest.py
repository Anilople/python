import threading
import time

def printTime(threadName,delay,delayCount = 10):
    while delayCount > 0:
        time.sleep(delay)
        delayCount -= 1
        print(threadName,time.ctime(time.time()),'---','delayCount:',str(delayCount))


threadLock = threading.Lock()
threads = []

class MyThread(threading.Thread):
    def __init__(self,threadID,name,delay,counter):
        threading.Thread.__init__(self)
        self.threadID =  threadID
        self.delay = delay
        self.name = name
        self.counter = counter
    def run(self):
        print("开始: "+self.name)
        print("获取锁")
        threadLock.acquire()
        printTime(self.name,self.delay,self.counter)
        threadLock.release()
        print("释放锁")
        print("退出: "+self.name)

# creat
thread1 = MyThread(1,'T-1',1,10)
thread2 = MyThread(2,'T-2',3,3)

threads = [thread1,thread2]

# begin
thread1.start()
thread2.start()
for thr in threads:
    thr.join()

print('quit')