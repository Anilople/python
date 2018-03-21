# coding: utf-8
import socket
import sys

def myEcho(sock,addr,dataSize = 1024):
    recv_data = sock.recv(dataSize) # 接收数据
    sock.sendto(recv_data,addr) # 发送数据回去
    recv_data = str(recv_data,encoding='utf8') # 解码
    print(recv_data) # 输出数据

sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP) # TCP

ip_port = ('127.0.0.1',6666) # 本机ip和端口,如果用户不输入自己的ip和端口,默认用这个
if len(sys.argv) >= 3: # 假设用户输入了正确的ip和端口
    ip_port = (sys.argv[1],int(sys.argv[2])) # 截取第2个元素当作ip,第3个当端口(强制转为数字), 因为第1个是程序名,不用

sk.bind(ip_port) # 绑定本地地址和端口

sk.listen(5) # 监听. 5 代表等待连接的最大数量

# 进入死循环,等待客户端连接
while True:
    print('waiting for client....')
    quitMark = False
    sock,addr = sk.accept() # 接收新连接
    print('new client connected')
    print('socket obj:',sock)
    print('client info:',addr)
    while True: # 死循环, 进入交互
        recv_data = sock.recv(1024) # 接受1024字节长度的数据, 返回类型为bytes
        recv_data = str(recv_data,encoding='utf8') # bytes -> str
        if recv_data == 'exit': # exit代表断开连接
            sock.close()
            print("disconnect, bye")
            break
        if recv_data == 'exitAll': # exitAll表示退出程序
            quitMark = True
            break
            
        print(recv_data) # 打印接收到的数据
        sock.sendto(bytes(recv_data,encoding='utf8'),addr) # 发送相同的数据给客户端

    if quitMark: # 退出标志
        print("quit while as you wish")
        break

sk.close() # 关闭socket

