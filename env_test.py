import gym
import gym_kiloBot
import time
import cv2
import numpy as np
def test1():
    print("This should run the env with out any render")
    env = gym.make("kiloBot-v0",n=10,screen_width=500,screen_heigth=500,render=False)
    env.reset()
    a = env.dummy_action(0.1,5)
    for i in range(1000):
        _,_,_,o=env.step([a]*env.n)
        out = o['critic_input']
        cv2.imshow("asdf",out)
        cv2.waitKey(10)
        if i%100==0:
            env.reset()
            time.sleep(0.05)
    #env.close()
    time.sleep(0.2)
    cv2.destroyAllWindows()
    return True

def test2():
    print("This should run the env with render")
    env = gym.make("kiloBot-v0",n=10,screen_width=500,screen_heigth=500,radius=5)
    env.reset()
    a = env.dummy_action(0.1,5)
    for i in range(1000):
        env.step([a]*env.n);env.render();
        if i%100==0:
            env.reset()
            time.sleep(0.05)
        time.sleep(0.01)
    env.reset()
    env.close()
    time.sleep(0.2)
    return True

def test3():
    print("This should run the env and be able to fetch all the graphs")
    env = gym.make("kiloBot-v0",n=10,screen_width=500,screen_heigth=500,radius=5)
    env.reset()
    a = env.dummy_action(0.1,5)
    for i in range(500):
        env.step([a]*env.n);env.render();
        env.graph_obj_distances()
        print(len(env.module_queue),"\n",env.module_queue)
        env.module_queue = []
        if i%100==0:
            env.reset()
            time.sleep(0.05)
        time.sleep(0.05)
    env.reset()
    env.close()
    time.sleep(0.2)
    return True

def test4():
    print("This should check for the histogram and the optimal region")
    env = gym.make("kiloBot-v0",n=10,screen_width=500,screen_heigth=500,radius=5)
    env.reset()
    a = env.dummy_action(0.1,5)
    for i in range(500):
        env.step([a]*env.n);env.render();
        env.graph_obj_distances()
        hist = env.fetch_histogram()
        print(hist,"\n",hist.shape)
        env.module_queue = []
        if i%100==0:
            env.reset()
            time.sleep(0.05)
        time.sleep(0.05)
    env.reset()
    env.close()
    time.sleep(0.2)
    return True

def test5():
    print("This should check for the localized target with and with the critic knowing (presence in the image)")
    env = gym.make("kiloBot-v0",n=10,objective="localization",screen_width=500,screen_heigth=500,radius=5)
    env.reset()
    a = env.dummy_action(0.1,5)
    for i in range(1000):
        observation,reward,done,info = env.step([a]*env.n);env.render();
        print(info["localization_bit"],"\n",info["target_distance"],"\n",info["neighbouring_bit"])
        if i%100==0:
            env.reset()
            time.sleep(0.05)
        time.sleep(0.01)
    env.reset()
    env.close()
    time.sleep(0.2)
    return True

if __name__=="__main__":
    testes = [test2,test3,test4,test5] ## dont have non render mode with render mode environement in the same session it causes a lot of problem
    print("I Have not completely designed the test to be user freind but just as a output of the env variable \n \t\t ^\_('_')_/^\n\n ");time.sleep(1)
    result = [0]*len(testes)
    for i,test in enumerate(testes):
        res = test()
        print("test"+str(i)+" - results "+("Passed!" if res else "Failed!"))
        result[i] = int(res)
    print("Passed "+str(sum(result))+"/"+str(len(testes))+" tests !")
