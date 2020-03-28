import argparse
import torch
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from .hdf5_load import bytes_to_tensor
from .argparse_util import *


class PushDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, seq_start=0, seq_length=20):
        self.datadir = datadir
        self.files = listdir_nohidden(datadir)
        self.seq_start = seq_start
        self.seq_length = seq_length

        self.file_seq_cts = []
        self.total_seq_ct = 0

        for file in self.files:
            with h5py.File(file, "r") as f:
                self.total_seq_ct += len(f)
                self.file_seq_cts.append(len(f))

        self.file_seq_cts = np.cumsum(self.file_seq_cts)

    def __len__(self):
        return self.total_seq_ct

    def __getitem__(self, index):
        # Index into the cumulative sum array to get the file
        file_index = np.argmax(self.file_seq_cts > index)
        if file_index == 0:
            seq_index = index
        else:
            seq_index = index - self.file_seq_cts[file_index - 1]

        with h5py.File(self.files[file_index], "r") as f:
            seq = f["trajectory_{:05d}".format(seq_index)]

            images = torch.stack(
                [
                    bytes_to_tensor(b)
                    for b in seq["images"][
                        self.seq_start : (self.seq_start + self.seq_length)
                    ]
                ]
            )
            states = torch.from_numpy(
                seq["states"][self.seq_start : (self.seq_start + self.seq_length)]
            )
            actions = torch.from_numpy(
                seq["actions"][self.seq_start : (self.seq_start + self.seq_length)]
            )

        return images, states, actions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datadir",
        type=dir_exists_read_privileges,
        help="Data file storage directory path",
    )

    args = parser.parse_args()

    ds = PushDataset(args.datadir)

    images, states, actions = ds.__getitem__(3)

    grid_img = make_grid(images, nrow=5)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
