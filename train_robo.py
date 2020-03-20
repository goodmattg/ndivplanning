import warnings

warnings.filterwarnings("ignore")

import argparse, os
import torch
import pdb
import diversity as div
import numpy as np
import os

from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from vis_tools import *

from models.action_decoder import Decoder
from models.action_discriminator import Discriminator

from torchvision.utils import save_image
from time import time
from diversity import VGG
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from utils.image_utils import *

from RoboNet.robonet.datasets.robonet_dataset_torch import RoboNetDataset
from RoboNet.robonet.datasets.robonet_dataset_torch import (
    worker_init_fn as robonet_worker_init_fn,
)

# Random Initialization
torch.manual_seed(1)
np.random.seed(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# Configurations and Hyperparameters
port_num = 8081
DEVICE = 1
lr_rate = 2e-4
num_epochs = 10000
num_sample = 6
noise_dim = 2
report_feq = 10
batch_size = 64
data_dir = "../hdf5"

hparams = {
    "color_augmentation": 0.3,  # std of color augmentation (set to 0 for no augmentations)
    "RNG": 0,
    "ret_fnames": True,
    "load_T": 0,
    "sub_batch_size": 8,
    "action_mismatch": 3,
    "state_mismatch": 3,
    "splits": [0.8, 0.1, 0.1],
    "same_cam_across_sub_batch": True,
}

dataset = RoboNetDataset(batch_size, data_dir, hparams=hparams)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=dataset._n_workers,
    worker_init_fn=robonet_worker_init_fn,
)

# Models
# TODO: Switch to not device
decoder, discriminator = (
    Decoder(noise_dim=noise_dim).to(DEVICE),
    Discriminator().to(DEVICE),
)

decoder.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
G_optimizer = optim.Adam(
    [{"params": decoder.parameters()}], lr=lr_rate, betas=(0.5, 0.999)
)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(0.5, 0.999))

step = 0
min_pred_error = np.inf
for epoch in range(num_epochs):

    for i, batch in enumerate(loader):
        ########## Inputs ########
        images, actions, states, _ = batch.values()

        images, actions, states = (
            images.squeeze(0).to(DEVICE).float(),
            actions.squeeze(0).to(DEVICE).float(),
            states.squeeze(0).to(DEVICE).float(),
        )

        images = norm(images)

        ########## Encode Current State ########
        noise = torch.randn(action.shape[0], 2).to(DEVICE)
        action_hat = decoder(torch.cat([feat_cur, noise], dim=1))

        ################## Train Discriminator ##################
        for _ in range(3):
            D_loss = nn.BCEWithLogitsLoss()(
                torch.squeeze(discriminator(action, feat_cur)),
                torch.ones(action.size(0)).to(DEVICE),
            ) + nn.BCEWithLogitsLoss()(
                torch.squeeze(discriminator(action_hat, feat_cur)),
                torch.zeros(action_hat.size(0)).to(DEVICE),
            )
            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()

        ########## G Loss ##########
        G_loss = nn.BCEWithLogitsLoss()(
            torch.squeeze(discriminator(action_hat, feat_cur)),
            torch.ones(action_hat.size(0)).to(DEVICE),
        )

        ########## Div Loss ##########
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        D_loss_np = D_loss.cpu().data.numpy()
        G_loss_np = G_loss.cpu().data.numpy()
        step += 1

        print(epoch, step, "D: ", D_loss_np, "G: ", G_loss_np)

        if step % report_feq == 0:
            state_cur_vis = denorm(state_cur[0]).detach().cpu().numpy().astype(np.uint8)
            action_gt = action[0].detach().cpu().numpy()
            action_hat = action_hat[0].detach().cpu().numpy()

            gt_color = (255, 0, 0)
            state_cur_arrow_gt = draw_action_arrow(state_cur_vis, action_gt, gt_color)

            hat_color = (0, 0, 255)
            state_cur_arrow_hat = draw_action_arrow(
                state_cur_vis, action_hat, hat_color
            )

            display.img_result(state_cur_arrow_gt, win=1, caption="state_cur_arrow_gt")
            display.img_result(
                state_cur_arrow_hat, win=2, caption="state_cur_arrow_hat"
            )

    if epoch % 1 == 0:
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(decoder, "models/decoder_" + str(epoch) + ".pt")
