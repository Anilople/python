from PIL import Image
import argparse

ascii_char=list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

# 将256灰度映射到字符集合上
def get_char(r,g,b,ascii_char,alpha=256):
    if alpha=0:
        return ' '
    length = len(ascii_char)
    gray = int(0.2126*r+0.7152*g+0.0722*b) # 心理学灰度公式
    
    unit = (256.0+1)/length
    return ascii_char[int(gray/unit)]

