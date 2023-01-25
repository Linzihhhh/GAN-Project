import torch
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(5,100),
            nn.ReLU(),
            nn.Linear(100,3000),
            nn.ReLU(),
            nn.Linear(3000,120000),
            nn.ReLU(),
            nn.Linear(120000,480000),
            )
    def forward(self,x):
        x=self.net(x)
        return x
        
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(480000,120000),
            nn.ReLU(),
            nn.Linear(120000,3000),
            nn.ReLU(),
            nn.Linear(3000,500),
            nn.ReLU(),
            nn.Linear(500,1)
            )
    def forward(self,x):
        x=self.net(x)
        return x

lr_G=0.001
lr_D=0.001
