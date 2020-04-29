import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import pdb
import torch
import sys
import numpy as np
import diversity as div
import logging

from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from vis_tools import *
from torchvision.utils import save_image
from utils.trajectory_loader import PushDataset
from time import time
from dotmap import DotMap

from argparse import ArgumentParser, ArgumentTypeError
from utils.cli_arguments.common_arguments import add_common_arguments
from utils.argparse_util import override_dotmap
from utils.file import make_paths_absolute
from utils.image_utils import norm, denorm


def fetch_push_control_evaluation(
    image_encoder: torch.nn.Module,
    fwd_model_autoencoder: torch.nn.Module,
    generator: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    config: DotMap,
):
    """Runs evaluation on the control task given models and a dataset
    
    Inputs:
        image_encoder: torch.nn.Module,
        fwd_model_autoencoder: torch.nn.Module,
        generator: torch.nn.Module,
        dataset: torch.utils.data.Dataset,

    Outputs:
        avg_action_error: float
        avg_image_loss: float
    """
    image_encoder.eval()
    fwd_model_autoencoder.eval()
    generator.eval()

    # Configurations and Hyperparameters
    random_seed = config.random_seed
    num_sample = config.evaluation.num_sample
    noise_dim = config.evaluation.noise_dim
    batch_size = config.evaluation.batch_size

    gpu_id = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        display = visualizer(port=config.log_port)

    # Random Initialization
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    def diverse_sampling(code):
        N, C = code.size(0), code.size(1)
        noise = torch.FloatTensor(N, num_sample, noise_dim).uniform_().to(gpu_id)
        code = (code[:, None, :]).expand(-1, num_sample, -1)
        code = torch.cat([code, noise], dim=2)
        return code, noise

    loader = data.DataLoader(
        dataset, batch_size=config.evaluation.batch_size, shuffle=False
    )

    # Initialize Loss
    l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()
    step = 0
    action_error_sum = 0

    for i, inputs in enumerate(loader):

        print("trajectory: ", i)
        images, states, actions, goal = inputs
        images, states, actions, goal = (
            images.float().to(gpu_id),
            states.float().to(gpu_id),
            actions.float().to(gpu_id),
            goal.float().to(gpu_id),
        )
        state_cur, state_target = torch.split(
            images, split_size_or_sections=[dataset.seq_length - 1, 1], dim=1
        )

        state_cur_gan = state_cur.reshape(-1, *(state_cur.size()[2:]))

        state_target_gan = torch.repeat_interleave(
            state_target.squeeze(dim=1), repeats=dataset.seq_length - 1, dim=0
        )
        actions_gan = actions[:, :-1].reshape(-1, actions.size()[-1])
        state_codes = image_encoder(state_cur_gan).detach()
        target_codes = image_encoder(state_target_gan).detach()

        codes = torch.cat([state_codes, target_codes], dim=1).squeeze()
        diverse_codes, noises = diverse_sampling(codes)
        diverse_codes, noises = diverse_codes[..., None, None], noises[..., None, None]

        actions = actions[:, :-1, :]

        action_hat = generator(diverse_codes.view(-1, diverse_codes.size(2)))
        action_hat = action_hat.view(batch_size, -1, 4)
        state_cur_fwd = state_cur[:, 0]
        image_error_sum = 0
        for image_num in range(dataset.seq_length - 1):

            if image_num != dataset.seq_length - 2:
                state_fut = state_cur[:, image_num + 1]
            else:
                state_fut = state_target

            state_fut_hat = fwd_model_autoencoder(
                state_cur_fwd, action_hat[:, image_num]
            )
            state_cur_fwd = state_fut_hat

            image_error = mse(state_fut_hat, state_fut)
            # Cumulative action error with diverse samples
            image_error_sum += image_error
            step += 1
        action_error = mse(
            torch.repeat_interleave(actions, repeats=num_sample, dim=1), action_hat
        )
        # Cumulative action error with diverse samples
        action_error_sum += action_error

    avg_action_error = action_error_sum / ((dataset.seq_length - 1) * len(loader))
    avg_image_loss = image_error_sum / ((dataset.seq_length - 1) * len(loader))

    # logging.info("Average action reconstruction loss:", avg_action_error)
    # logging.info("Average image loss", avg_image_loss)

    return avg_action_error.item(), avg_image_loss.item()


if __name__ == "__main__":

    parser = ArgumentParser(description="Interact with your training script")
    parser = add_common_arguments(parser)
    namespace = parser.parse_args()

    # Creates composite config from config file and CLI arguments
    config = override_dotmap(namespace, "config_file")
    # Converts all filepaths in keys ending with "_path" from relative to absolute filepath
    config = make_paths_absolute(os.getcwd(), config)

    ################################################
    # Load pretrained models and dataset loader
    ################################################
    gpu_id = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

    dataset = PushDataset(
        config.evaluation_data_path, seq_length=config.trajectory_length
    )

    image_encoder = torch.load(config.image_encoder_model_path, map_location=gpu_id)
    generator = torch.load(config.gan_decoder_model_path, map_location=gpu_id)

    fwd_model_autoencoder = torch.load(
        config.forward_model_autoencoder_path, map_location=gpu_id
    )

    ################################################
    # Run evaluation
    ################################################

    avg_action_error, avg_image_loss = fetch_push_control_evaluation(
        image_encoder, fwd_model_autoencoder, generator, dataset, config
    )
