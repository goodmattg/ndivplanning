import warnings
warnings.filterwarnings("ignore")
import argparse, os
import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from vis_tools import *
from model_utils import *
from torchvision.utils import save_image
import pdb
from time import time
from diversity import VGG
import diversity as div
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os

# Configurations and Hyperparameters
port_num = 8083
gpu_id = 1
lr_rate = 2e-4
num_epochs = 10000
num_sample = 6
noise_dim = 2
report_feq = 10

display = visualizer(port=port_num)

# Random Initialization
torch.manual_seed(1)
np.random.seed(1)

# noise sampling 
def noise_sampling(batch_size, num_sample, noise_dim):
    noise = torch.FloatTensor(batch_size, num_sample, noise_dim).uniform_().to(gpu_id)
    return noise

def denorm(tensor):
    return ((tensor+1.0)/2.0)*255.0

def norm(image):
    return (image/255.0-0.5)*2.0

def diverse_sampling(code):
    N, C = code.size(0), code.size(1)
    noise = torch.FloatTensor(N, num_sample, noise_dim).uniform_().to(gpu_id)
    code = (code[:,None,:]).expand(-1,num_sample,-1)
    code = torch.cat([code, noise], dim=2)
    return code, noise

##### helpers for visualization #####
img_px_size = 256
img_meter_size = 7.7
meter_to_px = img_px_size/img_meter_size

def world_to_image(p):
    p = meter_to_px * np.array([p[0], -1 * p[1]])
    p += np.array([img_px_size / 2.0, img_px_size / 2.0])
    return p

def init_hit_vector(angle, magnitude):
    return magnitude * np.array([np.cos(angle), np.sin(angle)])

def process_action(action):
    hit_px_coord = world_to_image((action[0], action[1]))
    angle = -1 * action[2]
    magnitude = meter_to_px * action[3]
    hit_vector = init_hit_vector(angle, magnitude)
    return hit_px_coord, hit_vector

def draw_action_arrow(state_cur_vis, action_gt, color=(255,0,0)):
    state_cur_np = np.swapaxes(state_cur_vis, 0, 2)
    state_cur_np = np.swapaxes(state_cur_np, 0, 1)

    hit_coord, hit_vec = process_action(action_gt)
    hit_coord, hit_vec = hit_coord/2, hit_vec/2
    start_point = (int(hit_coord[0]), int(hit_coord[1]))
    end_point = (int(hit_coord[0]+hit_vec[0]), int(hit_coord[1]+hit_vec[1]))

    state_cur_arrow = state_cur_np.copy()
    state_cur_arrow = cv2.arrowedLine(state_cur_arrow, start_point, end_point, color=color, tipLength = 0.5, thickness=1)
    state_cur_arrow = np.swapaxes(state_cur_arrow, 0, 2)
    state_cur_arrow = np.swapaxes(state_cur_arrow, 1, 2)
    return state_cur_arrow

# Dataloader
action_dir = '../data/conditional_roller_new/action/'
state_cur_dir = '../data/conditional_roller_new/state_cur/'
state_fut_dir = '../data/conditional_roller_new/state_fut/'
feat_cur_dir = '../data/conditional_roller_new/feat_cur/'
feat_fut_dir = '../data/conditional_roller_new/feat_fut/'
dataset = RopeData(action_dir, state_cur_dir, state_fut_dir, feat_cur_dir, feat_fut_dir)
loader = data.DataLoader(dataset, batch_size=64)

# Models
decoder = torch.load('models/decoder_73.pt').to(gpu_id)

for i in range(10):

    feat = np.load('../data/test_conditional_roller/'+str(i)+'_feat.npy')
    feat = torch.from_numpy(feat).unsqueeze(0).to(gpu_id)

    action_hat_arr_np = np.zeros((10000,4))
    for s in range(10000):

        ########## Encode Current State ########
        noise = torch.randn(1, 2).to(gpu_id)
        action_hat = decoder(torch.cat([feat, noise], dim=1))   

        action_hat_np = action_hat.cpu().data.numpy()
        action_hat_arr_np[s] = action_hat_np

        print(i, s)

    np.save('../cond_results/gan_'+str(i)+'.npy', action_hat_arr_np)

