### python3的socket通信

#### 使用TCP协议进行通信(近乎字典式的教程) 

一个程序为**服务端**,一个为**客户端**

对于**服务端**, 步骤如下:

```python
import socket # 载入socket的库
# 1. 建立一个socket
sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP)
# 参数意义
# AF_INET 代表用ipv4
# SOCK_STREAM 代表用tcp
# IPPROTO_TCP 代表使用的协议

# 2.将这个socket和ip以及端口进行绑定
sk.bind(('127.0.0.1',6666))
# 参数意义, 注意传入的是tuple
# 字符串'127.0.0.1'代表ip地址
# 数字6666 代表端口号

# 3.监听 -- listen, 看哪些客户端想和服务端进行通信
sk.listen(5)
# 参数意义
# 5 代表能支持连接的客户端数量(因为我们想一台主机能支持多个客户端与其连接), 但是要开多线程才可以支持1个以上,这里不用多线程

# 4.接收连接 -- accept (阻塞?)
sock, addr = sk.accept() # 一直等待连接
# 返回参数意义
# sock 为 一个socke对象, 可以用来收发信息
# addr 为 tuple, 和2中bind所用的信息类似

# 5.收发信息 -- 收用recv, 发用sendto
recv_data = sk.recv(1024) # 收信息
# 接收参数意义
# 	1024 代表一次接收的字节数, 如果客户端一次发送的信息超过这个数值,服务端就会报错
# 返回参数意义
#	recv_data 为收到的数据, 数据类型为bytes, 注意不是str(字符串)
sk.sendto(send_data,addr) # 发信息
# 参数意义
# send_data 为 bytes类型, 代表要发送的数据
# addr 为 tuple, 里边有客户端的ip地址和端口号, 类似('127.0.0.1',6666)


'''
辅助的工具
bytes 与 str 直接的转换
由于收发数据只能用bytes, 但是人在查看信息的时候, str更友好, 所以需要转换
bytes -> str : str(data,encoding='utf8')
str -> bytes : bytes(data,encoding='utf8)
'''
```

对于**客户端**:

```python
import socket # 载入socket的库
# 1. 建立一个socket
sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM,socket.IPPROTO_TCP)
# 参数意义
# AF_INET 代表用ipv4
# SOCK_STREAM 代表用tcp
# IPPROTO_TCP 代表使用的协议

# 2.连接服务器
sk.connect(('127.0.0.1',6666))
# 参数意义, 注意传入的是tuple
# 字符串'127.0.0.1'代表ip地址
# 数字6666 代表端口号

# 3.收发信息 -- 收用recv, 发用sendto
recv_data = sk.recv(1024) # 收信息
# 接收参数意义
# 	1024 代表一次接收的字节数, 如果客户端一次发送的信息超过这个数值,服务端就会报错
# 返回参数意义
#	recv_data 为收到的数据, 数据类型为bytes, 注意不是str(字符串)
sk.sendto(send_data,addr) # 发信息
# 参数意义
# send_data 为 bytes类型, 代表要发送的数据
# addr 为 tuple, 里边有客户端的ip地址和端口号, 类似('127.0.0.1',6666)


'''
辅助的工具
bytes 与 str 直接的转换
由于收发数据只能用bytes, 但是人在查看信息的时候, str更友好, 所以需要转换
bytes -> str : str(data,encoding='utf8')
str -> bytes : bytes(data,encoding='utf8)
'''
```

客户端直接connect就可以进行收发数据, 但是服务端由于要有同时为多个客户端服务的能力, 在程序上要复杂一些.