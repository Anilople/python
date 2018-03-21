# coding: utf-8
import socket
import sys

udpServer = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP) # UDP

ip_port = ('127.0.0.1',6666) # 本机ip和端口,如果用户不输入自己的ip和端口,默认用这个
if len(sys.argv) >= 3: # 假设用户输入了正确的ip和端口
    ip_port = (sys.argv[1],int(sys.argv[2])) # 截取第2个元素当作ip,第3个当端口(强制转为数字), 因为第1个是程序名,不用

udpServer.bind(ip_port) # 绑定本地地址和端口

print("use UDP, so there is no waiting for client...")
while True:
    recv_data, client_addr = udpServer.recvfrom(1024)
    print('-'*15) # newline
    print('new message from client:',client_addr)
    print(str(recv_data,encoding='utf8')) # 显示数据
    if str(recv_data,encoding='utf8') == 'exitAll': # exitAll代表断开连接并关闭服务端
        print("disconnect, bye")
        break
    udpServer.sendto(recv_data,client_addr) # echo 数据会客户端

udpServer.close() # 关闭socket

