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
        return feat_6, [feat_1, feat_2, feat_3, feat_4, feat_5]


class Decoder(nn.Module):
    # initializers
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(128 + 4, 1024, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024 + 1024, 512, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512 + 512, 256, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256 + 256, 128, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128 + 128, 64, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64 + 64, 32, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(32)
        self.conv_refine_1 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv_refine_1_bn = nn.BatchNorm2d(16)
        self.conv_refine_2 = nn.Conv2d(16, 3, 3, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z, feats):
        feat_1, feat_2, feat_3, feat_4, feat_5 = feats
        up_1 = F.relu(self.deconv1_bn(self.deconv1(z)))
        up_2 = F.relu(self.deconv2_bn(self.deconv2(torch.cat([up_1, feat_5], dim=1))))
        up_3 = F.relu(self.deconv3_bn(self.deconv3(torch.cat([up_2, feat_4], dim=1))))
        up_4 = F.relu(self.deconv4_bn(self.deconv4(torch.cat([up_3, feat_3], dim=1))))
        up_5 = F.relu(self.deconv5_bn(self.deconv5(torch.cat([up_4, feat_2], dim=1))))
        up_6 = F.relu(self.deconv6_bn(self.deconv6(torch.cat([up_5, feat_1], dim=1))))
        up_6 = F.relu(self.conv_refine_1_bn(self.conv_refine_1(up_6)))
        x_hat = F.tanh(self.conv_refine_2(up_6))
        return x_hat


if __name__ == "__main__":

    gpu_id = 1
    noise_dim = 16
    num_sample = 6
    decoder = Decoder().to(gpu_id)

    state_cur = torch.ones((1, 3, 128, 128)).to(gpu_id)
    encoder = Encoder().to(gpu_id)
    codes, feats = encoder(state_cur)

    action = torch.ones((1, 4)).to(gpu_id)

    state_action_codes = torch.cat([codes, action], dim=1)
    state_action_codes = state_action_codes.unsqueeze(2).unsqueeze(3)
    state_fut_hat = decoder(state_action_codes, feats)

    pdb.set_trace()
