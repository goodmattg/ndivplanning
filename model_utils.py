import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
from spectral_normalization import SpectralNorm

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # initializers
    def __init__(self, noise_dim):
        super(Decoder, self).__init__()

        # action decoding
        self.fc1 = nn.Linear(128+noise_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z):
        # action decoding 
        action = F.relu(self.fc1(z))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action_hat = self.fc4(action)
        return action_hat

class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(128+4, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, action, state_code):
        state_action = torch.cat([action, state_code], dim=1)
        h = F.leaky_relu(self.fc1(state_action))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        z = self.fc4(h)
        return z

def collapse_batch(batch):
    if len(batch.shape) == 3:
        _, _, C = batch.size()
        return batch.view(-1, C)
    elif len(batch.shape) == 5:
        _, _, C, H, W = batch.size()
        return batch.view(-1, C, H, W)
    else:
        print("Error: No need to collapse")
        return batch

def uncollapse_batch(batch):
    if len(batch.shape) == 2:
        N, C = batch.size()
        return batch.view(int(N/num_sample), num_sample, C)
    elif len(batch.shape) == 4:
        N, C, H, W = batch.size()
        return batch.view(int(N/num_sample), num_sample, C, H, W)
    else:
        print("Error: No need to un-collapse")
        return batch

# 30
def diverse_sampling(code):
    N, C = code.size(0), code.size(1)
    noise = torch.FloatTensor(N, num_sample, noise_dim).uniform_().to(gpu_id)
    code = (code[:,None,:]).expand(-1,num_sample,-1)
    code = torch.cat([code, noise], dim=2)
    return code, noise

if __name__ == '__main__':

    gpu_id = 1
    noise_dim = 2
    decoder, discriminator = Decoder(noise_dim=noise_dim).to(gpu_id), Discriminator().to(gpu_id)

    state_code = torch.ones((1,128)).to(gpu_id)
    noise = torch.ones((1,2)).to(gpu_id)
    action = torch.ones((1,4)).to(gpu_id)
    
    action_hat = decoder(torch.cat([state_code, noise], dim=1))
    
    discriminator = Discriminator().to(gpu_id)
    z = discriminator(action, state_code)
    # print('z: ', z.shape)
    
    pdb.set_trace()
