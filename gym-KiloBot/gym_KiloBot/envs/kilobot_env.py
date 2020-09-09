import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
import cv2
from kiloBot import KiloBot
import os

class KiloBotEnv(gym.Env):
    metadata={'render.modes':['human']}
    BLACK=(0,0,0);WHITE=(255,255,255)
    pygame.init()
    def __init__(self,n=5,module_color=(0,255,0),radius=5,screen_width=250,screen_heigth=250):
        super().__init__()     ##Check it  once never used before
        self.modules = []
        self.module_color = module_color
        self.screen_width = screen_width
        self.screen_heigth = screen_heigth
        self.screen = pygame.display.set_mode((self.screen_width,self.screen_heigth))
        pygame.display.set_caption("Swarm")
        for i in range(n):
            self.modules.append(KiloBot(module_color,
                                    radius,
                                    xinit=np.random.randint(0,screen_width-radius),
                                    yinit=np.random.randint(0,screen_heigth-radius),
                                    theta=(2*np.pi*np.random.random_sample()),
                                    screen_width=self.screen_width,
                                    screen_heigth=self.screen_heigth)
                                    )



    def step(self,actions):
        for module,action in zip(modules,actions):
            module.update(action)

    def reset(self):

        for module in modules:
            module.spawn()
    def render(self,mode='human',close=False):
        pass
