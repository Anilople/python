# -*- coding:utf-8 -*-

import socket

if __name__ == '__main__':
    PORT = 8000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', PORT))
    sock.listen(1)
    print 'Serving HTTP on port %s ...' % PORT

    while 1:
        conn, addr = sock.accept()
        print conn, addr
        request = conn.recv(1024)
        # HTTP响应消息
        response = "HTTP/1.1 200 OK\nContent-Type:text/html\nServer:myserver\n\nHello, World!"
        conn.sendall(response)
        conn.close()