## GYM-KiloBot

This is a environment which tries simulate many modular bots named kiloBot s , the environement is a Sub Class of OpenAI Gym Env so has a standard member functions.

### Usage

``` python
import gym
import gym_kiloBot

env = gym.make('kiloBot-v0',
        n=5,                    ## No of modules to be initiated
        objective = "graph"     ## This specifics the task can also be "localization"
        render=True,            ## If Passed through initializes a pygame display for output
        module_color=(0,255,0), ## This sets the color of each module
        radius = 5 ,            ## This sets the size of each module
        screen_width = 250,     ## Sets the width of the output
        screen_heigth = 250     ## Sets the heigth of the output
        )
```
This should initialize modules in a random configuration and initalizes the environment with given parameter.
``` python
action = env.dummy_action(theta=0.1,r=2)
actions = [action]*n
```
*dummy_action* is a basic class defination for the format for action of each module a list of object instances is to be passed into *step*
``` python
observation, reward, done, info = env.step(actions)
output = info['critic_input']
```
- observation is a vector of histograms for each modules for the graph task and is vector of distances for each modules from the target for the localization task.<br/>
- info contains a item with key 'critic_input' which contains the image that critic wants, so that headless training is possible.
- reward is based on the objective choosen refer the paper for that
- done its a continuing task so taking a break and reseting periodically is the best thing to do to sample as many states as posiible.

``` python
env.render()    ## updates the display only works if render is passed as True while initialization
env.reset()     ## Spawns each module onces and to get no error do reset once before use
```
These two functions works as expected nothing to explain here
``` python
env.close()    
```
Should close the rendering tab if open.<hr/>
**Do not try to use only one env in a script at a time as pygame server would be messed up if multiple instances are initialized**
<hr/>
