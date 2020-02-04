import torch
from torch.utils import data
from torchvision.transforms import functional as tvf
from torchvision.transforms import Compose, Resize, CenterCrop
import numpy as np
from PIL import Image
import pdb
import glob


class RopeData(data.Dataset):
    def __init__(
        self, action_dir, state_cur_dir, state_fut_dir, feat_cur_dir, feat_fut_dir
    ):

        n_len = 0
        for _ in glob.glob(str(action_dir) + "*"):
            n_len += 1

        action_list = []
        for i in range(n_len):
            action_list.append(str(action_dir) + str(i) + ".txt")

        state_cur_list = []
        for i in range(n_len):
            state_cur_list.append(str(state_cur_dir) + str(i) + ".png")

        state_fut_list = []
        for i in range(n_len):
            state_fut_list.append(str(state_fut_dir) + str(i) + ".png")

        feat_cur_list = []
        for i in range(n_len):
            feat_cur_list.append(str(feat_cur_dir) + str(i) + ".npy")

        feat_fut_list = []
        for i in range(n_len):
            feat_fut_list.append(str(feat_fut_dir) + str(i) + ".npy")

        self.action_list = action_list
        self.state_cur_list = state_cur_list
        self.state_fut_list = state_fut_list
        self.feat_cur_list = feat_cur_list
        self.feat_fut_list = feat_fut_list

    def __getitem__(self, idx):
        action = np.loadtxt(self.action_list[idx]).astype(np.float)
        state_cur = (
            np.array(Image.open(self.state_cur_list[idx]).resize((128, 128)))
            .transpose(2, 0, 1)
            .astype(np.float)
        )
        state_fut = (
            np.array(Image.open(self.state_fut_list[idx]).resize((128, 128)))
            .transpose(2, 0, 1)
            .astype(np.float)
        )
        feat_cur = np.load(self.feat_cur_list[idx])
        feat_fut = np.load(self.feat_fut_list[idx])
        return action, state_cur, state_fut, feat_cur, feat_fut

    def __len__(self):
        return len(self.action_list)


if __name__ == "__main__":
    action_dir = "../data/conditional_push/action/"
    state_cur_dir = "../data/conditional_push/state_cur/"
    state_fut_dir = "../data/conditional_push/state_fut/"
    feat_cur_dir = "../data/conditional_push/feat_cur/"
    feat_fut_dir = "../data/conditional_push/feat_fut/"
    dataset = RopeData(
        action_dir, state_cur_dir, state_fut_dir, feat_cur_dir, feat_fut_dir
    )
    loader = data.DataLoader(dataset, batch_size=10)
    for i, inputs in enumerate(loader):
        action, state_cur, state_fut, feat_cur, feat_fut = inputs
        print(i, action.shape, feat_cur.shape, feat_fut.shape)

        pdb.set_trace()

