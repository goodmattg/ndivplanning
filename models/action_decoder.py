import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Decoder(nn.Module):
    # initializers
    def __init__(self, noise_dim):
        super(Decoder, self).__init__()

        # action decoding
        self.fc1 = nn.Linear(128 + noise_dim, 64)
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
