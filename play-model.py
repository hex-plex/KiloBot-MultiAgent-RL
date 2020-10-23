from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras.layers import Input,Dense
from keras.utils import normalize
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import gym
import gym_kiloBot
import time
import os

FLAGS = flags.FLAGS
flags.DEFINE_integer("modules",10,"Defines the no of modules in the env")
flags.DEFINE_integer("time_steps",10000000,"This is the no of steps that the env would take before stoping")
flags.DEFINE_integer("histRange",10,"Defines the steps for the histograms")
flags.DEFINE_string("objective","graph","This defines which task is to be choosen")
flags.DEFINE_string("load_checkpoint",None,"specifies the location of the checkpoint to start training from")
flags.mark_flag_as_required('load_checkpoint')
hyperparam={
    'gamma':0.99, ## Mostly we can try using averaging rewards
    'actor_lr':1e-4,
    'critic_lr':1e-3,
    'lambda':0.65,
}

class ModelActor(tf.keras.Model):

    def __init__(self,input_dims,no_action=2):
        super().__init__()
        self.state_input=Input(shape=input_dims,name="state_input")
        self.fc1 = Dense(1024,activation='elu',name='forward1',kernel_initializer=keras.initializers.RandomUniform(minval=-1./1024,maxval=1./1024))
        self.fc2 = Dense(512,activation='elu',name='forward2',kernel_initializer=keras.initializers.RandomUniform(minval=-1./512,maxval=1./512))
        self.fc3 = Dense(256,activation='elu',name='forward3',kernel_initializer=keras.initializers.RandomUniform(minval=-1./256,maxval=1./256))
        self.fc4 = Dense(128,activation='elu',name='forward4',kernel_initializer=keras.initializers.RandomUniform(minval=-1./128,maxval=1./128))
        self.mu = Dense(no_action,activation='linear',name='mu1',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))
        self.sigma = Dense(no_action,activation='linear',name='sigma1',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))
    def call(self,input_data):
        #x = self.state_input(input_data)
        x = self.fc1(input_data)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        probmu = self.mu(x)
        probsigma = self.sigma(x)
        probsigma = tf.nn.softplus(probsigma) + 1e-5
        return probmu,probsigma

    def setup(self,gamma=0.99):
        self.gamma = gamma
        self.optimizer = Adam(lr=hyperparam['actor_lr'])

    def act(self,state):
        probmu, probsigma = self(np.array(state))
        dist = tfp.distributions.Normal(loc=probmu.numpy(),scale=probsigma.numpy())
        action = dist.sample([1])
        return action.numpy()
    def actor_loss(self,probmu,probsigma,actions,td):
        dist = tfp.distributions.Normal(loc=probmu,scale=probsigma)
        log_prob = dist.log_prob(actions + 1e-5)
        loss = -log_prob*td
        return loss
    def learn(self,prev_state,td):
        with tf.GradientTape() as tape:
            pm,ps = self(prev_state,training=True)
            action = self.act(prev_state)
            a_loss = self.actor_loss(pm,ps,action,td)
        grads = tape.gradient(a_loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return a_loss
def preprocessReplayBuffer(states,actions,rewards,gamma):
    discountedRewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
        sum_reward = r + gamma*sum_reward
        discountedRewards.append(sum_reward)
    discountedRewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(action, dtype=np.float32)
    discountedRewards = np.array(discountedRewards,dtype=np.float32)

    return states, actions, discountedRewards
def fetch_states_localize(observation,info,env):
    prev_state = np.append(np.array(info.get('localization_bit')).reshape(-1,1),
                            observation,axis=1)
    prev_state = np.append(np.array(info.get('target_distance')).reshape(-1,1),
                            prev_state,axis=1)
    prev_state = np.append(np.array(info.get('neighbouring_bit')).reshape(-1,1),
                            prev_state,axis=1)
    critic_prev_state = np.array([module.get_state(normalized=True) for module in env.modules],dtype=np.float32).reshape(1,-1)
    critic_prev_state = np.append(np.array((env.target[0]-env.screen_width/2,env.target[1]-env.sceen_height/2)).reshape(1,-1),critic_prev_state)
    return prev_state,critic_prev_state

def fetch_states_graph(observation,info,env):
    prev_state = observation
    critic_prev_state = np.array([module.get_state() for module in env.modules],dtype=np.float32).reshape(1,-1)
    return prev_state,critic_prev_state
def play(argv):
    env = gym.make("kiloBot-v0",
                    n=FLAGS.modules,
                    k=FLAGS.histRange,
                    render= True,
                    objective=FLAGS.objective,
                    screen_width=500,
                    screen_heigth=500,
                    )
    obj = False
    if FLAGS.objective=='localization':
        actor_model = ModelActor((None,env.k+3),no_action=2)
        obj = True
    else:
        actor_model = ModelActor((None,env.k),no_action=2)     ## This is just k hist features

    #actor_model.load_weights(os.getcwd()+"/"+FLAGS.load_checkpoint+"/actor_model.h5")
    iter = 0
    env.reset()         ## Doing this ensures the image feed has initialized
    a = env.dummy_action(0.1,5)
    observation,_,_,info = env.step([a]*env.n)
    if obj:
        fetch_states = fetch_states_localize
    else:
        fetch_states = fetch_states_graph
    prev_state,_ = fetch_states(observation,info,env)
    actor_model.setup(gamma=hyperparam['gamma'])
    actor_model.act(prev_state)
    actor_model.load_weights(os.getcwd()+"/"+FLAGS.load_checkpoint+"/actor_model.h5")
    best_reward = -100000
    while iter<FLAGS.time_steps:
        iter +=1
        env.render()
        action_inputs = np.squeeze(actor_model.act(prev_state))
        actions = []
        for action_input in action_inputs:
            actions.append(env.dummy_action(max(min(action_input[0],2*np.pi),-2*np.pi),max(min(action_input[1],10),0)))
        observation,reward,done,info = env.step(actions)
        state,_ = fetch_states(observation,info,env)
        prev_state = state
        best_reward = max(reward,best_reward)
        if iter%10000==0:
            env.reset()
            observation,_,_,info = env.step([a]*env.n)
            prev_state,_ = fetch_states(observation,info,env)
        if iter%100==0:
            print("iter "+str(iter)+" yeilds reward :"+str(reward))
        if done:
            env.reset()
            prev_state,critic_prev_state = fetch_states(observation,info,env)
    env.close()


if __name__=='__main__':
    app.run(play)
