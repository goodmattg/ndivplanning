import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


# G(z)
class Decoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 1024, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(128)

        self.deconv5 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(64)

        self.deconv6 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, z):
        up_1 = F.relu(self.deconv1_bn(self.deconv1(z)))
        up_2 = F.relu(self.deconv2_bn(self.deconv2(up_1)))
        up_3 = F.relu(self.deconv3_bn(self.deconv3(up_2)))
        up_4 = F.relu(self.deconv4_bn(self.deconv4(up_3)))
        up_5 = F.relu(self.deconv5_bn(self.deconv5(up_4)))
        up_6 = F.tanh(self.deconv6(up_5))
        return up_6
