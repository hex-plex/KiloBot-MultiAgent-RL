import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
import numpy as np
import gym
import gym_kiloBot

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs,2)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(),lr=0.001,momentum=0.9)
model_conv = train_model(model_conv,criterion,optimizer_conv,num_epochs=25)
