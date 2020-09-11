import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import numpy as np
import cv2
from .action import Action
from .arrow import *
from .kiloBot import KiloBot
import os

class KiloBotEnv(gym.Env):
    metadata={'render.modes':['human']}
    BLACK=(0,0,0);WHITE=(255,255,255);BLUE=(0,0,255);RED=(255,0,0)
    pygame.init()
    def __init__(self,n=5,k=5,objective="graph",render=True,dupper=None,dlower=None,dthreshold=None,sigma=None,module_color=(0,255,0),radius=5,screen_width=250,screen_heigth=250):
        super().__init__()     ##Check it  once never used before
        self.n = n
        self.k = k
        self.modules = []
        self.render_mode = render
        self.target = (0,0)
        if objective=="localization":
            self.obj = True
            self.target = (np.random.randint(0,screen_width-radius),np.random.randint(0,screen_heigth-radius))
        else:
            self.obj = False
        self.module_color = module_color
        self.screen_width = screen_width
        self.screen_heigth = screen_heigth
        self.target_color = self.BLUE
        self.relation_color = self.RED
        self.relationship_color = (255,0,0)
        self.radius = radius
        self.dummy_action = Action ## This is a class not a object that is stored
        self.module_queue = []
        self.graph_reward = 0
        self.target_reward = 0
        self.dupper = dupper or 14*self.radius
        self.dlower = dlower or 4*self.radius

        self.sigma = sigma or 0.025*self.screen_width
        if self.obj:
            self.dthreshold = dthreshold or 14*self.radius
        else:
            self.dthreshold = dthreshold or 16*self.radius
        self.ring_radius =  0.5*self.dthreshold
        self.epsilon = 1e-4
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
        self.arrow = init_arrow(self.module_color)
        self.action_space = spaces.Box(low = np.array([[0,0]]*self.n ,dtype=np.float32) ,
                                        high=np.array([[self.radius, 2*np.pi]]*self.n, dtype=np.float32))
        ### This will change with respect to output if its the histogram or the graph or the localization###
        ####################################################################################################
        self.observation_space = spaces.Box(low = np.zeros((self.n,self.k)) ,dtype=np.float32,
                                            high = 2*np.ones((self.n,self.k) , dtype=np.float32))

    def fetch_histogram(self):
        self.module_queue
        temphist = [ list([]) for i in range(self.n) ]  ## Dont use [[]]*self.n as it uses same pointer for all the lists and hence they are identical at the end
        stepsize = self.dthreshold/self.k
        steps = [i*stepsize for i in range(1,self.k+1)]
        for relation in self.module_queue:
            temphist[relation[0]].append(relation[2])
            temphist[relation[1]].append(relation[2])
        histvalues = []
        for histplot in temphist:
            histplot = np.array(histplot,dtype=np.float32)
            temp = []
            for step in steps:
                ans = np.sum(np.array(histplot<=self.dthreshold , dtype=np.float32)*(histplot*np.exp(-np.square(histplot - step)/(2*(self.sigma**2)))))
                temp.append(ans)
            temp = np.array(temp,dtype=np.float32)
            temp /= (np.sum(temp)+self.epsilon)
            histvalues.append(temp.copy())
            del temp
        return np.array(histvalues,dtype=np.float32)

    def graph_obj_distances(self):
        for i in range(self.n):
            for j in range(i+1,self.n):
                tempdst = (self.modules[i]-self.modules[j]).norm()
                if tempdst<=self.dthreshold:
                    self.module_queue.append([i,j,tempdst])
                    if self.dlower<=tempdst<=self.dupper:
                        self.graph_reward += tempdst/10
        return True

    def step(self,actions):
        if not pygame.display.get_init() and self.render_mode:
            raise Exception("Some problem in the rendering contiivity of the code OpenAI Wrapper messing it up! or try running reset once at the beginning")
        states=[]
        reward = 0
        self.screen.fill(self.BLACK)
        for module,action in zip(self.modules,actions):
            reward -= 0.05 * module.update(action)
            states.append(module.get_state())
            if (not self.obj) or (module.l!=1):
                pygame.draw.circle(self.screen,module.color,(module.rect.x,module.rect.y),module.radius)
            pygame.draw.line(self.screen,module.color,(module.rect.x,module.rect.y),
                                (module.rect.x + self.radius*2*np.cos(module.theta),module.rect.y + self.radius*2*np.sin(module.theta)),1)
            nar,nrect = rotate_arrow(self.arrow.copy(),
                            (module.rect.x + self.radius*2*np.cos(module.theta),module.rect.y + self.radius*2*np.sin(module.theta)),
                            module.theta)
            self.screen.blit(nar,(nrect.x,nrect.y))
            pygame.draw.circle(self.screen,(0,102,51),(module.rect.x,module.rect.y),int(self.ring_radius),2)## Draw A circle around it and draw the Region of interest
        self.graph_obj_distances()
        if self.obj:
            mask = [0]*self.n
            pygame.draw.circle(self.screen,self.target_color,self.target,self.radius) ## draw  the blue dot
            for i,module in enumerate(self.modules):
                if module.dist(self.target).norm()<=self.dthreshold:
                    mask[i] = 1
                if module.dist(self.target).norm()<=5*self.radius or module.l==1:
                    module.l=1
                    pygame.draw.circle(self.screen,(255,0,0),(module.rect.x,module.rect.y),module.radius)
                    self.target_reward += 1
            reward += self.target_reward
            neighbouring_bit = [0]*self.n
            for relation in self.module_queue:
                if relation[2]<=self.dthreshold:
                    if mask[relation[0]]==1:
                        neighbouring_bit[relation[1]]=1
                    if mask[relation[1]]==1:
                        neighbouring_bit[relation[0]]=1
        else:
            reward += self.graph_reward
            for relation in self.module_queue:
                if relation[2]<=self.dthreshold:
                    i ,j = relation[:2]
                    pygame.draw.line(self.screen,(255,0,0),self.modules[i].get_state()[:2],self.modules[j].get_state()[:2])
        hist = self.fetch_histogram()
        self.module_queue = []
        done = False
        critic_input = np.array(pygame.surfarray.array3d(self.screen).swapaxes(0,1),dtype=np.uint8).reshape([self.screen_width,self.screen_heigth,3])
        info = {"critic_input":critic_input,"localization_bit": [module.l for module in self.modules]}
        if self.obj:
            info["target_distance"]= [ module.dist(self.target).norm() if module.dist(self.target).norm()<self.dthreshold else -1 for module in self.modules]
            info["neighbouring_bit"] = neighbouring_bit
        self.graph_reward,self.target_reward = 0,0
        return hist,reward,done,info


    def reset(self):
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.screen_width,self.screen_heigth))
            pygame.display.set_caption("Swarm")
        else:
            self.screen = pygame.Surface((self.screen_width,self.screen_heigth))
        self.screen.fill(self.BLACK)
        if not pygame.display.get_init():
            pygame.display.init()
        for module in self.modules:
            module.spawn()
        if self.obj:
            self.target = (np.random.randint(self.radius,self.screen_width-self.radius),np.random.randint(self.radius,self.screen_heigth-self.radius))
            pygame.draw.circle(self.screen,self.target_color,self.target,self.radius)
    def render(self,mode='human',close=False):
        if not pygame.display.get_init() and self.render_mode:
            self.screen = pygame.display.set_mode((self.screen_width,self.screen_heigth))
            pygame.display.set_caption("Swarm")
            pygame.draw.circle(self.screen,module.color,(module.rect.x,module.rect.y),module.radius)
        elif not self.render_mode:
            raise Exception("You cant render if you have passed its arguement as False")
        pygame.display.flip()
        if mode=="human":
            self.clock.tick(60)

    def close(self):
        if self.render_mode:
            pygame.display.quit()
        pygame.quit()
