import warnings

# import sys
# ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

# if ros_path in sys.path:
#     sys.path.remove(ros_path)

warnings.filterwarnings("ignore")
import argparse
import torch
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import logging

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
from models.forward_encoder import ForwardAutoencoder

from argparse import ArgumentParser, ArgumentTypeError
from utils.cli_arguments.common_arguments import add_common_arguments
from utils.argparse_util import override_dotmap
from utils.file import make_paths_absolute
from utils.image_utils import norm, denorm


def train(config):

    random_seed = config.random_seed
    lr_rate = config.training.forward.learning_rate
    num_epochs = config.training.forward.num_epochs
    report_feq = config.training.forward.report_feq
    batch_size = config.training.forward.batch_size
    epochs_per_stage = config.training.forward.epochs_per_stage
    gamma = config.training.forward.step_lr_gamma

    gpu_id = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

    # Random Initialization
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    display = visualizer(port=config.log_port)

    # Dataloader
    dataset = PushDataset(config.train_data_path, seq_length=config.trajectory_length)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    forward_autoencoder = ForwardAutoencoder().to(gpu_id)
    forward_autoencoder.decoder.weight_init(mean=0.0, std=0.02)
    forward_autoencoder.encoder.weight_init(mean=0.0, std=0.02)

    # Initialize Loss
    mse = nn.MSELoss()

    # Initialize Optimizer
    optimizer = optim.Adam(
        [
            {"params": forward_autoencoder.decoder.parameters()},
            {"params": forward_autoencoder.encoder.parameters()},
        ],
        lr=lr_rate,
        betas=(0.5, 0.999),
    )

    torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_per_stage, gamma=gamma)

    step = 0
    min_pred_error = np.inf
    for epoch in range(num_epochs):
        loss_np_sum = 0

        for i, inputs in enumerate(loader):
            images, _, actions, _ = inputs
            images, actions = images.to(gpu_id), actions.to(gpu_id)

            for image_num in range(dataset.seq_length - 1):
                state_cur = images[:, image_num]
                state_fut = images[:, image_num + 1]

                state_fut_resid_hat = forward_autoencoder(
                    state_cur, actions[:, image_num]
                )

                loss = mse(state_fut_resid_hat, state_fut - state_cur)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                loss_np = loss.cpu().data.numpy()
                loss_np_sum += loss_np

            if step % report_feq == 0:

                state_cur_vis = [
                    denorm(state_cur[0]).detach().cpu().numpy().astype(np.uint8)
                ]
                state_fut_vis = [
                    denorm(state_fut[0]).detach().cpu().numpy().astype(np.uint8)
                ]
                state_fut_resid_hat_vis = [
                    denorm(state_fut_resid_hat[0])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                ]
                state_fut_hat_vis = [
                    denorm(state_fut_resid_hat[0] + state_cur[0])
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                ]
                display.img_result(state_cur_vis, win=1, caption="state_cur_vis")
                display.img_result(state_fut_vis, win=2, caption="state_fut_vis")
                display.img_result(
                    state_fut_resid_hat_vis, win=3, caption="state_fut_resid_hat_vis"
                )
                display.img_result(
                    state_fut_hat_vis, win=4, caption="state_fut_hat_vis"
                )

        # Log average epoch loss
        avg_loss = loss_np_sum / ((dataset.seq_length - 1) * len(loader))
        display.plot("loss", "train", "Forward Model Loss", epoch, avg_loss)

        logging.info(
            "{}, {}: reconstruction loss per epoch: {}".format(epoch, step, avg_loss)
        )

        if epoch % epochs_per_stage == epochs_per_stage - 1:

            if not os.path.exists(config.forward_save_path):
                os.makedirs(config.forward_save_path)

            torch.save(
                forward_autoencoder,
                os.path.join(
                    config.forward_save_path,
                    "forward_autoencoder_{}.pt".format(str(epoch)),
                ),
            )


if __name__ == "__main__":

    parser = ArgumentParser(description="Interact with your training script")
    parser = add_common_arguments(parser)
    args = parser.parse_args()

    # Creates composite config from config file and CLI arguments
    config = override_dotmap(args, "config_file")
    # Converts all filepaths in keys ending with "_path" from relative to absolute filepath
    config = make_paths_absolute(os.getcwd(), config)

    train(config)
