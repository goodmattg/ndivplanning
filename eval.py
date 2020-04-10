import warnings

warnings.filterwarnings("ignore")
import argparse
import os
import pdb
import torch

import numpy as np
import diversity as div

from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
from data_utils import *
from vis_tools import *
from torchvision.utils import save_image
from utils.trajectory_loader import PushDataset
from time import time
import sys
from models.forward_encoder import Decoder, Encoder

# Configurations and Hyperparameters
port_num = 8085
# gpu_id = 'cpu'
gpu_id = 1
lr_rate = 2e-4
num_epochs = 100
num_sample = 1
noise_dim = 2
report_feq = 10
batch_size = 8

# display = visualizer(port=port_num)

# Random Initialization
torch.manual_seed(1)
np.random.seed(1)

def diverse_sampling(code):
    N, C = code.size(0), code.size(1)
    noise = torch.FloatTensor(N, num_sample, noise_dim).uniform_().to(gpu_id)
    code = (code[:, None, :]).expand(-1, num_sample, -1)
    code = torch.cat([code, noise], dim=2)
    return code, noise


def denorm(tensor):
    return ((tensor + 1.0) / 2.0) * 255.0


def norm(image):
    return (image / 255.0 - 0.5) * 2.0

dataset = PushDataset("128_128_data", seq_length=16)
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loading all the required Models : image_autoencoder, 
# image_encoder = torch.load("Models/image_autoencoder/encoder.pt",map_location=torch.device('cpu'))
image_encoder = torch.load("Models/image_autoencoder/encoder.pt").to(gpu_id)
image_encoder.eval()

# fwd_model_encoder = torch.load("Models/forward_model/forward_encoder.pt",map_location=torch.device('cpu'))
fwd_model_encoder = torch.load("Models/forward_model/forward_encoder.pt").to(gpu_id)
fwd_model_encoder.eval()

fwd_model_decoder = torch.load("Models/forward_model/forward_decoder.pt").to(gpu_id)
fwd_model_decoder.eval()

generator = torch.load("Models/gan/gan_decoder.pt").to(gpu_id)
generator.eval()

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()
step = 0
action_error_sum = 0
for i, inputs in enumerate(loader):
    images,states,actions = inputs
    images, states, actions = (
            images.float().to(gpu_id),
            states.float().to(gpu_id),
            actions.float().to(gpu_id),
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

    actions = actions[:,:-1,:]

    action_hat = generator(diverse_codes.view(-1, diverse_codes.size(2)))
    action_hat = action_hat.view(batch_size,-1,4)
    state_cur_fwd = state_cur[:,0]
    image_error_sum = 0
    print(actions.shape,action_hat.shape)
    for image_num in range(dataset.seq_length-1):
        if image_num!=dataset.seq_length-2:
            state_fut = state_cur[:,image_num+1]
        else:
            state_fut = state_target
        code_fwd, feats_fwd = fwd_model_encoder(state_cur_fwd)
        print(action_hat.shape)
        state_action_concate = torch.cat([code_fwd, action_hat[:,image_num]], dim=1)
        state_action_concate = state_action_concate.unsqueeze(2).unsqueeze(3)
        state_fut_hat = fwd_model_decoder(state_action_concate, feats_fwd)
        state_cur_fwd = state_fut_hat

        image_error = mse(state_fut_hat,state_fut)
        image_error_sum += image_error
        step += 1

    if step%report_feq==0:
        state_cur_vis = [
                denorm(state_cur_fwd[0]).detach().cpu().numpy().astype(np.uint8)
            ]
        state_fut_vis = [
            denorm(state_fut[0]).detach().cpu().numpy().astype(np.uint8)
        ]
        state_fut_hat_vis = [
            denorm(state_fut_hat[0]).detach().cpu().numpy().astype(np.uint8)
        ]
        # display.img_result(state_cur_hat_vis, win=1, caption="state_cur_vis")
        # display.img_result(state_fut_vis, win=2, caption="state_fut_vis")
        # display.img_result(state_fut_hat_vis, win=3, caption="state_fut_hat_vis")
    action_error = mse(actions,action_hat)
    action_error_sum += action_error
avg_action_error = action_error_sum / ((dataset.seq_length - 1) * len(loader))
avg_image_loss = image_error_sum / ((dataset.seq_length - 1) * len(loader))
print("Average reconstruction loss:", avg_action_error)
print("Average image loss", avg_image_loss)

    


# EVALUATION WIHOUT FEEDBACK
# 1. Load the dataset, load models: Encoder Action generator Forward_model
# 2. separate the images from the target image
# 3. Encode the state image (0 to the second last image) -> pass images to the encoder
# 4. Generate actions: pass noise + encoding of target + encoding of state to the generator 
# 5. Pass action+state to forward model
# 6. Get the next stage image
# 7. Go to 3. {repeat for trajectory_length-1}
# 8. Compare pixel wise distance b/w target and generated target image
# 8. OR Compare action sequence with original actions


#spectral normalization, lower number of 










# EVALUATION WITH TRUE FEEDBACK -> DISCUSS FURTHER
# 1. Load the dataset, load models: Encoder Action generator Forward_model
# 2. separate the images from the target image
# 3. Encode the state image (0 to the second last image) -> pass images to the encoder
# 4. Generate and record actions: pass noise + encoding of target + encoding of state to the generator 
# 5. Pass action+state to forward model
# 6. Get the next stage image
# 7. Go to 3. {repeat for trajectory_length-1}
# 8. Calculate mean diff b/w Values of actions at each stage with ground truth
