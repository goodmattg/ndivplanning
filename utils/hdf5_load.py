import h5py
import io
import torch

from PIL import Image
from torchvision.transforms import ToTensor


def bytes_to_tensor(raw_bytes):
    """Use PIL to load raw JPEG compressed bytes, then convert to torch tensor"""
    return ToTensor()(Image.open(io.BytesIO(raw_bytes)))


def load_hdf5(file, seq_length, seq_start):

    return images, states, actions, camera_transform, camera_intrinsic

    raise NotImplementedError
