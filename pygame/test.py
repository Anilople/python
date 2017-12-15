import pygame,sys
import time
import random
pygame.init()
screencaption=pygame.display.set_caption('hello world')
screen=pygame.display.set_mode([640,480])
screen.fill([255,255,255])
pygame.draw.circle(screen,[255,0,0],[50,50],30,1)
pygame.draw.rect(screen,[0,255,0],[50,50,100,200],3)
dx=0
dy=0
points=[[50+dx,50+dy],[70+dx,90+dy],[100+dx,100+dy]]
pygame.draw.lines(screen,[0,0,255],True,points,2)
    # pygame.draw.rect(screen,[255,0,0],[left,top,width,height],3)
pygame.display.flip()

while True:
    for event in pygame.event.get():
        points=[[50+dx,50+dy],[70+dx,90+dy],[100+dx,100+dy]]
        pygame.draw.lines(screen,[0,0,255],True,points,2)
        pygame.display.flip()
        dx=dx+10
        dy=dy+10
        print dx,dy
        pygame.time.delay(200)
        pygame.display.flip()
        if event.type==pygame.QUIT:
            sys.exit()
