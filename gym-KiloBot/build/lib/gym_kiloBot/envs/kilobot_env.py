import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
import cv2
from .kiloBot import KiloBot
import os

class KiloBotEnv(gym.Env):
    metadata={'render.modes':['human']}
    BLACK=(0,0,0);WHITE=(255,255,255)
    pygame.init()
    def __init__(self,n=5,objective="graph",module_color=(0,255,0),radius=5,screen_width=250,screen_heigth=250):
        super().__init__()     ##Check it  once never used before
        self.n = n
        self.modules = []
        if objective=="localization":
            self.obj = True
        else:
            self.obj = False
        self.module_color = module_color
        self.screen_width = screen_width
        self.screen_heigth = screen_heigth
        self.screen = pygame.display.set_mode((self.screen_width,self.screen_heigth))
        pygame.display.set_caption("Swarm")
        self.screen.fill(self.BLACK)
        self.radius = radius
        for i in range(n):
            self.modules.append(KiloBot(module_color,
                                    radius,
                                    xinit=np.random.randint(0,screen_width-radius),
                                    yinit=np.random.randint(0,screen_heigth-radius),
                                    theta=(2*np.pi*np.random.random_sample()),
                                    screen_width=self.screen_width,
                                    screen_heigth=self.screen_heigth)
                                    )
        self.clock = pygame.time.Clock()
        self.action_space = spaces.Box(low = np.array([[0,0]]*self.n ,dtype=np.float32) ,
                                        high=np.array([[self.radius, 2*np.pi]]*self.n, dtype=np.float32))
        ### This will change with respect to output if its the histogram or the graph or the localization###
        ####################################################################################################
        self.observation_space = spaces.Box(low = np.array([[0, 0, 0]]*self.n ,dtype=np.float32),
                                            high = np.array([[self.screen_width, self.screen_heigth, 2*np.pi ]]*self.n , dtype=np.float32))

    def fetch_histogram(self,states):
        pass

    def graph_obj_distances(self):
        pass

    def step(self,actions):
        if not pygame.display.get_init():
            raise Exception("Some problem in the rendering contiivity of the code OpenAI Wrapper messing it up!")
        states=[]
        reward = 0
        self.screen.fill(self.BLACK)
        for module,action in zip(self.modules,actions):
            reward -= 0.05 * module.update(action)
            states.append(module.get_state())
            pygame.draw.circle(self.screen,module.color,module.rect)
            ## Draw a arrow for the same
            ## Draw A circle around it and draw the Region of interest
        if self.obj:
            pygame.draw.circle(self.screen,self.BLUE,self.target) ## draw  the blue dot
        else:
            pass ## Draw the relationship joints also
        done = False
        critic_input = np.array(pygame.PixelArray(self.screen),dtype=np.uint8).T.reshape([self.screen_width,self.screen_heigth,1])
        info = {"critic_input":critic_input}
        return states,reward,done,info


    def reset(self):
        self.screen.fill(self.BLACK)
        if not pygame.display.get_init():
            pygame.display.init()
        for module in self.modules:
            module.spawn()
            pygame.draw.circle(self.screen,module.color,module.rect)
        if self.obj:
            self.target = (np.random.randint(self.radius,self.screen_width-self.radius),np.random.randint(self.radius,self.screen_heigth-self.radius))
            pygame.draw.circle(self.screen,self.BLUE,self.target)
    def render(self,mode='human',close=False):
        pygame.display.flip()
        if mode=="human":
            self.clock.tick(60)

    def close(self):
        pygame.display.quit()
        return True
