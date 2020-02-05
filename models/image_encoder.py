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
