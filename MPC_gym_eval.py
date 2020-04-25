# eval_MPC

import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import gym
import pdb
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np
import diversity as div
import logging
from PIL import Image


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

def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def render(env):
    """Directs OpenAI gym environment to render to numpy RGB array"""
    return env.render(mode="rgb_array")

def image_from_state(state_cur_mpc,i,image_num):
    temp = state_cur_mpc[0].permute(1,2,0)
    img_numpy = temp.numpy()
    img_numpy = (img_numpy - np.min(img_numpy))/(np.max(img_numpy)-np.min(img_numpy))
    # print(np.min(img_numpy))
    file_name = "results/"+str(i)+"result"+str(image_num)+".png"
    plt.imsave(file_name,img_numpy)

def get_state(env,args):
    image = render(env)
    im = Image.fromarray(image)
    im = im.resize(args.image_shape, Image.LANCZOS)
    _image = np.array(im)
    # plt.imsave("temp2.png",_image)
    state_0 = torch.from_numpy(_image)
    state_0 = state_0[np.newaxis, :] 
    state_0 = state_0.permute(0, 3, 1, 2)
    return norm(state_0)

def controlled_reset(env,states,goal):
    
    observation = env.reset()
    object_qpos = env.sim.data.get_joint_qpos("object0:joint")
    assert object_qpos.shape == (7,)
    object_qpos[:2] = np.array(states)[0][0][3:5]
    env.env.sim.data.set_joint_qpos("object0:joint", object_qpos)
    env.sim.forward()
    env.env.goal = np.array(goal).ravel()
    return env

def fetch_push_control_evaluation(
    args,
    image_encoder: torch.nn.Module,
    fwd_model_autoencoder: torch.nn.Module,
    generator: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    config: DotMap,
    env
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

    #Getting the first state from gym environment
    
  
    image_encoder.eval()
    fwd_model_autoencoder.eval()
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
    goal_error = []
    for i, inputs in enumerate(loader):
        images, states, actions, goal = inputs
        images, states, actions, goal = (
            images.float().to(gpu_id),
            states.float().to(gpu_id),
            actions.float().to(gpu_id),
            goal.float().to(gpu_id),
        )
        # print("goal",goal)

        env = controlled_reset(env,states,goal)
        # obss = env.env._get_obs()
        # print("desired goal",obss["desired_goal"])
        # print("env.sim.data.get_joint_qpos",env.sim.data.get_joint_qpos("object0:joint")[:3])
        
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
            
            for ts in range(Th): #min(Th,dataset.seq_length -1 -image_num)
                
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
                # print(action_now_hat.size())
                if ts==0:
                    action_now_taken = action_now_hat


                state_fut_hat = fwd_model_autoencoder(state_cur_fwd,action_now_hat.squeeze(1))
                state_cur_fwd = state_fut_hat
                
            best_act_ind = 0
            for ro in range(rollouts):
                err_now = mse(state_fut_hat[ro],state_target[0])
                if err_now<min_error:
                    min_error = err_now
                    best_act_ind = ro
            best_action_so_far = action_now_taken[best_act_ind]
            take_action = best_action_so_far.detach().numpy().ravel()
            env.step(take_action)
            best_action_list.append(best_action_so_far)
            
            state_cur_mpc = get_state(env,args)
            image_from_state(state_cur_mpc,i,image_num)
            image_error = mse(state_cur_mpc, state_fut)

            # Cumulative action error with diverse samples
            image_error_sum += image_error
            step += 1
        
        obss = env.env._get_obs()
        DG = obss["desired_goal"] # desired goal position
        OP = env.sim.data.get_joint_qpos("object0:joint")[:3] # object position
        dist_to_goal = np.sum((DG - OP)**2)**0.5
        # print("desired goal",DG)
        # print("env.sim.data.get_joint_qpos",OP)
        print("results of trajectory ", i+1)
        print("distance from the goal: ",dist_to_goal)
        goal_error.append(dist_to_goal)
        goal_error_arr = np.asarray(goal_error)
        success_rate = np.sum(goal_error_arr<config.evaluation.threshold)/len(goal_error_arr)
        print("success rate so far: ", success_rate)
        action_hat = torch.cat(best_action_list,dim=0)
        action_error = mse(
            torch.repeat_interleave(actions, repeats=num_sample, dim=1), action_hat
        )
        # Cumulative action error with diverse samples
        action_error_sum += action_error
        # print(action_error_sum)
    avg_goal_error = sum(goal_error)/len(goal_error)
    avg_action_error = action_error_sum / ((dataset.seq_length - 1) * len(loader))
    avg_image_loss = image_error_sum / ((dataset.seq_length - 1) * len(loader))
    goal_error_arr = np.asarray(goal_error)
    success_rate = np.sum(goal_error_arr<config.evaluation.threshold)/len(goal_error_arr)
    # logging.info("Average reconstruction loss:", avg_action_error)
    # logging.info("Average image loss", avg_image_loss)

    return avg_action_error.item(), avg_image_loss.item(), avg_goal_error,success_rate



if __name__ == "__main__":

    parser = ArgumentParser(description="Interact with your training script")
    parser = add_common_arguments(parser)



    parser.add_argument(
            "--image-shape",
            nargs=2,
            type=int,
            default=(128, 128),
            help="Output image shape (WIDTH, HEIGHT) via PIL.Image.resize()",
        )


    args = parser.parse_args()
    namespace = parser.parse_args()

    # Creates composite config from config file and CLI arguments
    config = override_dotmap(namespace, "config_file")
    # Converts all filepaths in keys ending with "_path" from relative to absolute filepath
    config = make_paths_absolute(os.getcwd(), config)

    ################################################
    # Load pretrained models and dataset loader
    ################################################
    gpu_id = torch.device(config.gpu_id if torch.cuda.is_available() else "cpu")

    dataset = PushDataset(config.evaluation_data_path, seq_length==config.trajectory_length)
    image_encoder = torch.load(config.image_encoder_model_path, map_location=gpu_id)
    generator = torch.load(config.gan_decoder_model_path, map_location=gpu_id)

    fwd_model_encoder = torch.load(
        config.forward_model_encoder_path, map_location=gpu_id
    )
    fwd_model_decoder = torch.load(
        config.forward_model_decoder_path, map_location=gpu_id
    )
    fwd_model_autoencoder = torch.load(
        config.forward_model_autoencoder_path, map_location=gpu_id
    )
    ######################################################################
    # Create and configure the environment
    ######################################################################
    env = gym.make("FetchPush-v1")

    env.target_range = 0.30
    env.obj_range = 0.30

    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {
        "obs": observation["observation"].shape[0],
        "goal": observation["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
    }
    _ = render(env)

    env.viewer.cam.distance = 1.0
    env.viewer.cam.azimuth = 130
    env.viewer.cam.elevation = -40.0


    ################################################
    # Run evaluation
    ################################################

    avg_action_error, avg_image_loss, avg_goal_error, success_rate = fetch_push_control_evaluation(
        args,image_encoder, fwd_model_autoencoder, generator, dataset, config,env
    )
