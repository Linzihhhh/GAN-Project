import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pydub

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.net=nn.Sequential(
            nn.Linear(5,100),
            nn.ReLU(),
            nn.Linear(100,3000),
            nn.ReLU(),
            nn.Linear(3000,1200),
            nn.LeakyReLU(),
            nn.Linear(1200,80000)
            )
        self.RNN=nn.RNN(80000,20,5,batch_first=True)
    def forward(self,x):
        x=x.to(torch.float32)
        x=self.net(x)
        x=self.RNN(x)
        return x
        
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.RNN=nn.RNN(80000,20,5,batch_first=True)
        self.net=nn.Sequential(
            nn.Linear(80000,1200),
            nn.LeakyReLU(),
            nn.Linear(1200,3000),
            nn.LeakyReLU(),
            nn.Linear(3000,500),
            nn.LeakyReLU(),
            nn.Linear(500,1)
            )
    def forward(self,x):
        x=x.to(torch.float32)
        x=self.RNN(x)
        x=self.net(x)
        return x

lr_G=0.001
lr_D=0.001

generator=Generator()
discriminator=Discriminator()

optimizer_G=torch.optim.RMSprop(generator.parameters(),lr=lr_G)
optimizer_D=torch.optim.RMSprop(discriminator.parameters(),lr=lr_D)

data=np.load('Dataset/batched piano data.npy')
print(data.shape)


print(data.shape)
p=0
Epochs=10
for i in range(Epochs):
    for x in data:
        real_sound=Variable(torch.FloatTensor(x))
        optimizer_D.zero_grad()
        z=Variable(torch.from_numpy(np.random.normal(0,1,(5,5))))
        fake_sound=generator(z).detach()
        
        loss_d= -torch.mean(discriminator(real_sound))+torch.mean(discriminator(fake_sound))
        
        loss_d.backward()
        optimizer_D.step()
        p=p+1
        if(p%100==0):
            optimizer_G.zero_grad()
            gen_sound=generator(z)
            loss_G=-torch.mean(discriminator(gen_sound))
            loss_G.backward()
            optimizer_G.step()
            song=gen_sound.data.numpy()
            
            
            write(f'{i}th {p}step.mp3',8000,song[0])
            print("Step=",p)
            break
torch.save(generator,'G.pt')
torch.save(discriminator,'D.pt')




        