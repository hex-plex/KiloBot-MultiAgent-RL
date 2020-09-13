from absl import app
from absl import flags
import tensorflow as tf
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
def model_critic(input_dims):
    state_input=Input(shape=input_dims,name="state_input")
    x = Dense(512,activation='elu',name='forward1',kernel_initializer=keras.initializers.RandomUniform(minval=-1./512,maxval=1./512))(state_input)
    x = Dense(256,activation='elu',name='forward2',kernel_initializer=keras.initializers.RandomUniform(minval=-1./256,maxval=1./256))(x)
    x = Dense(128,activation='elu',name='forward3',kernel_initializer=keras.initializers.RandomUniform(minval=-1./128,maxval=1./128))(x)
    value_func = Dense(128,activation='linear',name='value_func',kernel_initializer=keras.initializers.RandomUniform(minval=-3e-4,maxval=3e-4))(x)

    model = Model(inputs=[state_input],outputs=[value_func])
    model.compile(optimizer=Adam(lr=hyperparam['critic_lr']),loss='mse')
    model.summary()
    return model

def model_actor(input_dims):
    state_input=Input(shape=input_dims,name="state_input")
    x = Dense(1024,activation='elu',name='forward1',kernel_initializer=keras.initializers.RandomUniform(minval=-1./1024,maxval=1./1024))(state_input)
    x = Dense(512,activation='elu',name='forward2',kernel_initializer=keras.initializers.RandomUniform(minval=-1./512,maxval=1./512))(x)
    x = Dense(256,activation='elu',name='forward3',kernel_initializer=keras.initializers.RandomUniform(minval=-1./256,maxval=1./256))(x)
    x = Dense(128,activation='elu',name='forward4',kernel_initializer=keras.initializers.RandomUniform(minval=-1./128,maxval=1./128))(x)
    policy = None  ## Left it empty to be sure as td error is to be input

    model = Model(inputs=[state_input],outputs=[policy])
    model.compile(optmizer=Adam(lr=hyperparam['actor_lr']),loss=[actor_loss()])
    model.summary()
    return model
    
def main(argv):
    pass

if __name__=='__main__':
    app.run(main)
