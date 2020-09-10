import pygame
import numpy as np

def init_arrow(color):
    arrow = pygame.Surface((4,4))
    arrow.fill((0,0,0))
    pygame.draw.line(arrow,color,(0,0),(2,2))
    pygame.draw.line(arrow,color,(0,4),(2,2))
    arrow.set_colorkey((0,0,0))
    return arrow

def rotate_arrow(arrow,pos,angle):
    nar = pygame.transform.rotate(arrow,np.degrees(angle))
    nrect = nar.get_rect(center=pos)
    return nar,nrect
