# KiloBot-MultiAgent-RL
This is an experimentation to learn about Swarm Robotics with help of MultiAgent Reinforcement learning. We have used KiloBot as a platform as these are very simple in the actions space and have very high degree of symmetry. The Main inspiration of this project is this paper[[1]](#1)

## Setting Up
``` bash
git clone https://github.com/hex-plex/KiloBot-MultiAgent-RL
cd KiloBot-MultiAgent-RL
pip install --upgrade absl-python \
                      tensorflow \
                      gym \
                      opencv-python \
                      tensorflow_probability \
                      keras \
                      pygame
pip install -e gym-kiloBot
```
This should fetch and install the basics packages needed and should install the environment
### Sample of environment
These envs are running on a constant dummy actions

|**Env with Graph Objective**|Env with Localization Objective|
|--|--|
|![Output-1](/env_test_graph_compress.gif?raw=true)|![Output-2](/env_test_localize_compress.gif?raw=true)|

## Usage
``` bash
python env-test.py ## This will help you check the functionality of the environement and should give the sample code to understand the apis as well.
python model-train.py \
        --headless=True \             ## for headless training default False
        --objective="localization" \  ## defines the objective default is graph
        --modules=10 \                ## This defines the no of modules to be initialized default 10
        --time_steps=100000 \         ## This is the total no of steps the agent will take while learning
        --histRange=10 \              ## This is the no of mu values for the histograms
        --logdir="logs" \             ## This specifies the log location for TensorBoard
        --checkpoints="checkpoints"   ## This is for defining the location where the model is to be saved
        --load-checkpoints="checkpoints/iter-500" ## This loads the specified iteration

python play-model.py ## This should load trained weights and show the performance
```
## Training
This is underconstruction hope to see you on the other side ...... :smile:
## References
<a id="1">[1]</a>
**Guided Deep Reinforcement Learning for Swarm Systems** [[arXiv:1709.06011v1]](https://arxiv.org/abs/1709.06011) [cs.MA] 18 Sep 2017
