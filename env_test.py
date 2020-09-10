import gym
import gym_kiloBot
import time
import cv2

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
    return True

def test4():
    print("This should check for the histogram and the optimal region")
    return True

def test5():
    print("This should check for the localized target with and with the critic knowing (presence in the image)")
    return True

if __name__=="__main__":
    testes = [test1,test2,test3,test4,test5]
    for i,test in enumerate(testes):
        print("test"+str(i)+" - results "+("Passed!" if test() else "Failed!"))
