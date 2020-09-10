import gym
import gym_kiloBot



def test1():
    print("This should run the env with out any render")
    env = gym.make("kiloBot-v0",n=5,render=False)
    env.close()

def test2():
    print("This should run the env with render")
    
def test3():
    print("This should run the env and be able to fetch all the graphs")

def test4():
    print("This should check for the histogram and the optimal region")

def test5():
    print("This should check for the localized target with and with the critic knowing (presence in the image)")


if __name__=="__main__":
    testes = [test1,test2,test3,test4,test5]
    for i,test in enum(testes):
        print("test"+str(i)+" - results "+("Passed!" if test() else "Failed!"))
