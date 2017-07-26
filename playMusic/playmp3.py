# coding=utf-8
import time
import pygame
import mp3play

# f='C:Users/lambda/Music/Taylor Swift - Love Story.mp3' # windows test

f='Taylor Swift - Love Story.mp3' # windows test

clip=mp3play.load(f)
clip.play()
time.sleep(20)
clip.stop()

pygame.mixer.init()
print('播放泰勒')
track=pygame.mixer.music.load(f)

pygame.mixer.music.play() # 播放
time.sleep(10)
pygame.mixer.music.stop() # 停止

