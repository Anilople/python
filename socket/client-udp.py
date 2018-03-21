# coding: utf-8
import socket
import sys

udpClient = socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP) # UDP

ip_port = ('127.0.0.1',6666) # 本机ip和端口,如果用户不输入自己的ip和端口,默认用这个
if len(sys.argv) >= 3: # 假设用户输入了正确的ip和端口
    ip_port = (sys.argv[1],int(sys.argv[2])) # 截取第2个元素当作ip,第3个当端口(强制转为数字), 因为第1个是程序名,不用

while True: # 不断的读取用户的输入并发送给服务端
    inputData = input("send: ")
    udpClient.sendto(bytes(inputData,encoding='utf8'),ip_port)
    if inputData[:4] == 'exit': # 退出
        break
    else:
        recv_data,server_addr = udpClient.recvfrom(1024)
        recv_data = str(recv_data,encoding='utf8')
        print('message from server:',server_addr)
        print(recv_data) # 输出服务端数据
        print('-'*15) # newline

udpClient.close() # 关闭socket
print('udp close.')

