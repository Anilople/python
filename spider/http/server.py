#coding=utf-8

import socket

s=socket.socket()
host=socket.gethostname()
print host
port = 12345

s.bind((host,port))

s.listen(5)

while True:
    c,addr=s.accept()
    print "address: ",addr
    c.send("great")
    c.close()