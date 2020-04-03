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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(1024, 128, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        # print('===== Encoder =====')
        feat_1 = F.relu(self.conv1_bn(self.conv1(x)))
        # print('feat_1: ', feat_1.shape)
        feat_2 = F.relu(self.conv2_bn(self.conv2(feat_1)))
        # print('feat_2: ', feat_2.shape)
        feat_3 = F.relu(self.conv3_bn(self.conv3(feat_2)))
        # print('feat_3: ', feat_3.shape)
        feat_4 = F.relu((self.conv4(feat_3)))
        # print('feat_4: ', feat_4.shape)
        feat_5 = F.relu((self.conv5(feat_4)))
        # print('feat_5: ', feat_5.shape)
        feat_6 = self.conv6(feat_5)
        # print('feat_6: ', feat_6.shape)
        feat_6 = feat_6.view(feat_6.size(0), -1)
        # print('feat_6: ', feat_6.shape)
        return feat_6


class Decoder(nn.Module):
    # initializers
    def __init__(self, noise_dim):
        super(Decoder, self).__init__()

        # action decoding
        self.fc1 = nn.Linear(256 + noise_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, 4)

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
        action = F.relu(self.fc4(action))
        action_hat = self.fc5(action)
        return action_hat


class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1)
        self.img_encoder = Encoder()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, action, state):
        state_code = self.img_encoder(state)
        state_action = torch.cat([action, state_code], dim=1)
        h = F.leaky_relu(self.fc1(action))
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
        return batch.view(int(N / num_sample), num_sample, C)
    elif len(batch.shape) == 4:
        N, C, H, W = batch.size()
        return batch.view(int(N / num_sample), num_sample, C, H, W)
    else:
        print("Error: No need to un-collapse")
        return batch


if __name__ == "__main__":

    gpu_id = 1
    noise_dim = 16
    num_sample = 6
    decoder, discriminator = (
        Decoder(noise_dim=noise_dim).to(gpu_id),
        Discriminator().to(gpu_id),
    )

    x = torch.ones((2, 3, 128, 128)).to(gpu_id)
    encoder = Encoder().to(gpu_id)
    codes = encoder(x)

    N, C, H, W = x.shape
    x_unsqueeze = (x[:, None, :]).expand(-1, num_sample, C, H, W)
    x_unsqueeze = x_unsqueeze.contiguous().view(
        -1, x_unsqueeze.shape[2], x_unsqueeze.shape[3], x_unsqueeze.shape[4]
    )

    diverse_codes, noises = diverse_sampling(codes)
    diverse_codes, noises = diverse_codes[..., None, None], noises[..., None, None]

    action_hat = decoder(diverse_codes.view(-1, diverse_codes.size(2)))

    discriminator = Discriminator().to(gpu_id)
    z = discriminator(action_hat, x_unsqueeze)

    print("z: ", z.shape)

    pdb.set_trace()
