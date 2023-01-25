import torch
import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
    def forward(self,x):
        
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(480000,120000),
            nn.ReLU()
            nn.Linear(120000,3000),
            nn.ReLU()
            nn.Linear(3000,500),
            nn.RELU()
            nn.Linear(500,1)
            )
        
        
    def forward(self,x):
        
        