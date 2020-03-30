import warnings

warnings.filterwarnings("ignore")
import argparse
import torch
import os
import numpy as np
import pdb

from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from vis_tools import *

# from model_utils import *
from torchvision.utils import save_image
from torchvision import transforms
from time import time
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR
from utils.trajectory_loader import PushDataset
from models.image_autoencoder import Decoder, Encoder

# Configurations and Hyperparameters
port_num = 8082
gpu_id = 1
lr_rate = 2e-4
num_epochs = 3
num_sample = 6
noise_dim = 2
report_feq = 10

display = visualizer(port=port_num)

# Random Initialization
torch.manual_seed(1)
np.random.seed(1)


def denorm(tensor):
    return ((tensor + 1.0) / 2.0) * 255.0


def norm(image):
    return (image / 255.0 - 0.5) * 2.0


# Dataloader
dataset = PushDataset("128_128_data")
loader = data.DataLoader(dataset, batch_size=16, shuffle=True)

# Models
encoder = Encoder().to(gpu_id)
decoder = Decoder().to(gpu_id)
decoder.weight_init(mean=0.0, std=0.02)
encoder.weight_init(mean=0.0, std=0.02)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
optimizer = optim.Adam(
    [{"params": decoder.parameters()}, {"params": encoder.parameters()}],
    lr=lr_rate,
    betas=(0.5, 0.999),
)

step = 0
min_pred_error = np.inf
for epoch in range(num_epochs):
    for i, inputs in enumerate(loader):
        ########## Inputs ########
        images, _, _ = inputs
        images = images.to(gpu_id)

        # Flatten image trajectories to image batch
        state_cur = images.view(-1, *(images.size()[2:]))
        # state_cur = norm(state_cur)

        z = encoder(state_cur)
        state_cur_hat = decoder(z)

        recon_loss = mse(state_cur_hat, state_cur)

        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()

        step += 1

        recon_loss_np = recon_loss.cpu().data.numpy()
        print(epoch, step, "recon_loss_np: ", recon_loss_np)

        if step % report_feq == 0:
            state_cur_vis = denorm(state_cur[0]).detach().cpu().numpy().astype(np.uint8)
            state_cur_hat_vis = (
                denorm(state_cur_hat[0]).detach().cpu().numpy().astype(np.uint8)
            )

            display.img_result(state_cur_vis, win=1, caption="state_cur_vis")
            display.img_result(state_cur_hat_vis, win=2, caption="state_cur_hat_vis")

    if epoch % 1 == 0:
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(encoder, "models/encoder_" + str(epoch) + ".pt")
        torch.save(decoder, "models/decoder_" + str(epoch) + ".pt")
