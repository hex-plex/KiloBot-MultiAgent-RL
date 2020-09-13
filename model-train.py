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


FLAGS = flags.FLAGS
flags.DEFINE_boolean("headless",False,"False to render the environment")
flags.DEFINE_integer("modules",10,"Defines the no of modules in the env")
flags.DEFINE_string("objective","graph","This defines which task is to be choosen")
flags.DEFINE_string("logdir","logs","Defines the logging directory")
flags.DEFINE_string("checkpoints","checkpoints","Defines the directory where model checkpoints are to be saved or loaded from")
flags.DEFINE_string("load-checpoint",None,"specifies the location of the checkpoint to start training from")
hyperparam={
    'gamma':0.99, ## Mostly we can try using averaging rewards
    'actor_lr':1e-4,
    'critic_lr':1e-3,
    'lambda':0.65,
}
def actor_loss():
    def loss(y_true,y_pred):
        return 0
    return loss
class model_critic(tf.keras.Model):

    def __init__(self,input_dims):
        super().__init__()
        self.state_input=Input(shape=input_dims,name="state_input")
        self.f1 = Dense(512,activation='elu',name='forward1',kernel_initializer=keras.initializers.RandomUniform(minval=-1./512,maxval=1./512))
        self.f2 = Dense(256,activation='elu',name='forward2',kernel_initializer=keras.initializers.RandomUniform(minval=-1./256,maxval=1./256))
        self.f3 = Dense(128,activation='elu',name='forward3',kernel_initializer=keras.initializers.RandomUniform(minval=-1./128,maxval=1./128))
        self.value_func = Dense(128,activation='linear',name='value_func',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-4,maxval=3e-4))

    def call(self,input_data):
        x = self.state_input(input_data)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        v = self.value_func(x)
        return v

    def setup(self,gamma=0.99):
        self.gamma = gamma
        self.optimizer = Adam(lr=hyperparam['critic_lr'])

    def learn(self,reward,prev_state,state,done):
        with tf.GradientTape() as tape:
            v_1 = self(prev_state,training=True)
            v = self(state,training=True)
            td = reward + self.gamma*v(1-np.array(done,dtype=np.int)) - v_1
            c_loss = td**2
        grads = tape.gradient(c_loss,self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return c_loss

class model_actor(tf.keras.Model):

    def __init__(self,input_dims,no_action):
        super().__init__()
        self.state_input=Input(shape=input_dims,name="state_input")
        self.fc1 = Dense(1024,activation='elu',name='forward1',kernel_initializer=keras.initializers.RandomUniform(minval=-1./1024,maxval=1./1024))
        self.fc2 = Dense(512,activation='elu',name='forward2',kernel_initializer=keras.initializers.RandomUniform(minval=-1./512,maxval=1./512))
        self.fc3 = Dense(256,activation='elu',name='forward3',kernel_initializer=keras.initializers.RandomUniform(minval=-1./256,maxval=1./256))
        self.fc4 = Dense(128,activation='elu',name='forward4',kernel_initializer=keras.initializers.RandomUniform(minval=-1./128,maxval=1./128))
        self.mu = Dense(no_action,activation='linear',name='mu1',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))
        self.sigma = Dense(no_action,activation='linear',name='sigma1',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-3,maxval=3e-3))
    def call(self,input_data):
        x = self.state_input(input_data)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.fc4(x)
        probmu = self.mu(x)
        probsigma = self.sigma(x)
        probsigma = tf.nn.softplus(probsigma) + 1e-5
        return probmu,probsigma

    def setup(self,gamma=0.99):
        self.gamma = gamma
        self.optimizer = Adam(lr=hyperparam['actor_lr'])

    def act(self,state):
        probmu, probsigma = self(np.array[state])
        dist = tfp.distributions.Normal(loc=probmu.numpy(),scale=probsigma.numpy())
        action = dist.sample([1])
        return action.numpy()
    def actor_loss(self,probmu,probsigma,actions,td):


def main(argv):
    pass

if __name__=='__main__':
    app.run(main)
