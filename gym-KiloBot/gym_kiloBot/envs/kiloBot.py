import pygame
from random import radint
import numpy as np
BLACK = (0,0,0)
class KiloBot(pygame.sprite.Sprite):

    def __init__(self,color,radius,xinit=None,yinit=None,theta=None,screen_width,screen_heigth):
        super().__init__()
        self.image = pygame.Surface([2*radius,2*radius])
        self.image.fill(BLACK)
        self.radius = radius
        self.color = color
        self.screen_width = screen_width
        self.screen_heigth = screen_heigth
        pygame.draw.circle(self.image, self.color,(self.radius,self.radius),self.radius)
        self.rect = self.image.get_rect()
        if (xinit is None) or  (yinit is None) or (theta is None):
            self.spawn()
    def update(self,action):
        self.theta += action.theta
        self.theta %= 2*np.pi
        self.rect.x = self.rect.x + action.r*(np.cos(self.theta))
        self.rect.x %= self.screen_width
        self.rect.y = self.rect.y+action.r*(np.sin(self.theta))
        self.rect.y %= self.screen_heigth
        return action.r
    def get_state(self):
        return (self.rect.x,self.rect.y,self.theta)
    def spawn(self):
        self.rect.x = randint(0,self.screen_width-1)
        self.rect.y = randint(0,self.screen_heigth-1)
        self.theta = 2*np.pi* np.random.random_sample()
        return True
