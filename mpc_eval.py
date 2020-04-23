# eval_MPC

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


def denorm(tensor):
    return ((tensor + 1.0) / 2.0) * 255.0


def norm(image):
    return (image / 255.0 - 0.5) * 2.0


def fetch_push_control_evaluation(
    image_encoder: torch.nn.Module,
    fwd_model_encoder: torch.nn.Module,
    fwd_model_decoder: torch.nn.Module,
    generator: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    config: DotMap,
):
    """Runs evaluation on the control task given models and a dataset
    
    Inputs:
        image_encoder: torch.nn.Module,
        fwd_model_encoder: torch.nn.Module,
        fwd_model_decoder: torch.nn.Module,
        generator: torch.nn.Module,
        dataset: torch.utils.data.Dataset,

    Outputs:
        avg_action_error: float
        avg_image_loss: float
    """
    image_encoder.eval()
    fwd_model_encoder.eval()
    fwd_model_decoder.eval()
    generator.eval()

    # Configurations and Hyperparameters
    random_seed = config.random_seed
    num_sample = config.evaluation.num_sample
    noise_dim = config.evaluation.noise_dim
    batch_size = config.evaluation.batch_size
    rollouts = config.mpc.rollouts
    Th = config.mpc.time_horizon
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
        images, states, actions, goal = inputs
        images, states, actions, goal = (
            images.float().to(gpu_id),
            states.float().to(gpu_id),
            actions.float().to(gpu_id),
            goal.float().to(gpu_id),
        )
        print(dataset.seq_length)
        state_cur, state_target = torch.split(
            images, split_size_or_sections=[dataset.seq_length - 1, 1], dim=1
        )

        actions = actions[:,:-1,:]

        # print(state_cur_fwd.size())
        image_error_sum = 0
        best_action_list = []
        for image_num in range(dataset.seq_length - 1):
            print(image_num)
            if image_num != dataset.seq_length - 2:
                state_fut = state_cur[:, image_num + 1]
            else:
                state_fut = state_target

            if image_num==0:
                state_cur_mpc = state_cur[:,0]

            min_error = 10000000000
            
            state_cur_fwd = state_cur_mpc
            state_cur_fwd = state_cur_fwd.repeat(rollouts,1,1,1)
            print("ppppp", state_cur_fwd.size())
            
            for ts in range(min(Th,dataset.seq_length -1 -image_num)):
                
                # getting action
                state_now = state_cur_fwd
                target_now = state_target.squeeze(1)
                
                target_now = target_now.repeat(rollouts,1,1,1)
            
                state_now_codes = image_encoder(state_now).detach()
                target_now_codes = image_encoder(target_now).detach()
                now_codes = torch.cat([state_now_codes, target_now_codes], dim=1).squeeze()
                
                diverse_now_codes, now_noises = diverse_sampling(now_codes)
                diverse_now_codes, now_noises = diverse_now_codes[..., None, None], now_noises[..., None, None]
                action_now_hat = generator(diverse_now_codes.view(-1, diverse_now_codes.size(2)))
                action_now_hat = action_now_hat.view(rollouts,-1,4)
                print(action_now_hat.size())
                if ts==0:
                    action_now_taken = action_now_hat

                #forward model
                code_fwd, feats_fwd = fwd_model_encoder(state_cur_fwd)
                state_action_concate = torch.cat([code_fwd, action_now_hat.squeeze(1)], dim=1)
                state_action_concate = state_action_concate.unsqueeze(2).unsqueeze(3)
                state_fut_hat = fwd_model_decoder(state_action_concate, feats_fwd)
                state_cur_fwd = state_fut_hat
            
            best_act_ind = 0
            for ro in range(rollouts):
                err_now = mse(state_fut_hat[ro],state_target[0])
                if err_now<min_error:
                    min_error = err_now
                    best_act_ind = ro
            best_action_so_far = action_now_taken[best_act_ind]
            
            code_fwd, feats_fwd = fwd_model_encoder(state_cur_mpc)
            state_action_concate = torch.cat([code_fwd, best_action_so_far.squeeze(1)], dim=1)
            state_action_concate = state_action_concate.unsqueeze(2).unsqueeze(3)
            state_cur_mpc = fwd_model_decoder(state_action_concate, feats_fwd)
            best_action_list.append(best_action_so_far)

            image_error = mse(state_cur_mpc, state_fut)

            # Cumulative action error with diverse samples
            image_error_sum += image_error
            step += 1
        action_hat = torch.cat(best_action_list,dim=1)
        action_error = mse(
            torch.repeat_interleave(actions, repeats=num_sample, dim=1), action_hat
        )
        # Cumulative action error with diverse samples
        action_error_sum += action_error
        print(action_error_sum)
    avg_action_error = action_error_sum / ((dataset.seq_length - 1) * len(loader))
    avg_image_loss = image_error_sum / ((dataset.seq_length - 1) * len(loader))

    # logging.info("Average reconstruction loss:", avg_action_error)
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

    dataset = PushDataset(config.evaluation_data_path, seq_length=8)

    image_encoder = torch.load(config.image_encoder_model_path, map_location=gpu_id)
    generator = torch.load(config.gan_decoder_model_path, map_location=gpu_id)

    fwd_model_encoder = torch.load(
        config.forward_model_encoder_path, map_location=gpu_id
    )
    fwd_model_decoder = torch.load(
        config.forward_model_decoder_path, map_location=gpu_id
    )

    ################################################
    # Run evaluation
    ################################################

    avg_action_error, avg_image_loss = fetch_push_control_evaluation(
        image_encoder, fwd_model_encoder, fwd_model_decoder, generator, dataset, config
    )
