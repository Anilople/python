import _thread
import time
# _thread.start_new_thread ( function, args[, kwargs] )
# function : the function for a thread
# args : a tuple
# kwargs : 可选参数

def printTime(threadName,delay,delayCount = 10):
    while delayCount > 0:
        time.sleep(delay)
        delayCount -= 1
        print(threadName,time.ctime(time.time()),'---','delayCount:',str(delayCount))

try:
    _thread.start_new_thread(printTime,('T1',1))
    _thread.start_new_thread(printTime,('T2',3))
except:
    print('Fail to creat thread')

while True:
    time.sleep(1)