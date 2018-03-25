# 读方法
# .read() 返回str, 文件的所有内容
# .read(size) 返回str, 每次返回size个字节

# 写方法
# .write(str) 写入str, 覆盖原有文件内容
fileName = 'fileIO/file.txt'

with open(fileName,'a') as f:
    print(f.read())