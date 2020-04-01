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
from PIL import Image

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
from models.forward_encoder import Decoder, Encoder
# Configurations and Hyperparameters
port_num = 8082
gpu_id = 1
lr_rate = 2e-4
num_epochs = 50
num_sample = 6
noise_dim = 2
report_feq = 10

torch.manual_seed(1)
np.random.seed(1)

# Dataloader
# gpu_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = PushDataset("data")
loader = data.DataLoader(dataset, batch_size=8, shuffle=True)

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
        images, _, actions = inputs
        loss_np_sum = 0
        for image_num in range(actions.shape[1] -1):
            state_cur = images[:,image_num]
            state_fut = images[:,image_num+1]
            
            code,feats = encoder(state_cur)
            state_action_concate = torch.cat([code,actions[:,image_num]],dim=1)
            state_action_concate = state_action_concate.unsqueeze(2).unsqueeze(3)
            state_fut_hat = decoder(state_action_concate,feats)

            loss = mse(state_fut_hat,state_fut)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step +=1
            loss_np = loss.cpu().data.numpy()
            loss_np_sum += loss_np

        avg_loss = loss_np_sum/(actions.shape[1]-1)
        print(epoch,step,"reconstruction loss per batch:", avg_loss)
        
        err_dict = {"loss_np:",avg_loss}
        display.plot_error(error_dict)

        if step % report_feq == 0:
            state_cur_vis = [denorm(state_cur[0]).detach().cpu().numpy().astype(np.uint8)]
            state_fut_vis = [denorm(state_fut[0]).detach().cpu().numpy().astype(np.uint8)]
            state_fut_hat_vis = [denorm(state_fut_hat[0]).detach().cpu().numpy().astype(np.uint8)]
            display.img_result(state_cur_vis, win=1, caption="state_cur_vis")
            display.img_result(state_fut_vis, win=2, caption="state_fut_vis")
            display.img_result(state_fut_hat_vis, win=3, caption="state_fut_hat_vis")

        if epoch%10 == 9:
            if not os.path.exists('models'):os.makedirs('models')
            torch.save(encoder, 'models/encoder_'+str(epoch)+'.pt')
            torch.save(decoder, 'models/decoder_'+str(epoch)+'.pt')      



