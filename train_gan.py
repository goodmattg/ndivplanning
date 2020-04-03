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
from utils.trajectory_loader import PushDataset
from models.image_autoencoder import Encoder
from models.gan import Decoder, Discriminator
from torchvision.utils import save_image
from time import time

from torch.optim.lr_scheduler import StepLR

# Configurations and Hyperparameters
port_num = 8085
gpu_id = 1
lr_rate = 2e-4
num_epochs = 100
num_sample = 6
noise_dim = 2
report_feq = 10
batch_size = 8
# Number of discriminator steps per generator step
discr_steps_per_gen = 3
# Number of training stages
epochs_per_stage = 25

display = visualizer(port=port_num)

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


# Dataloader
# gpu_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = PushDataset("128_128_data", seq_length=16)
loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Use pre-trained encoder
encoder = Encoder().to(gpu_id)
encoder.load_state_dict(torch.load("models/autoencoder_pretrained/encoder.pt"))
encoder.eval()

decoder = Decoder(noise_dim=noise_dim).to(gpu_id)
discriminator = Discriminator().to(gpu_id)

decoder.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

# Initialize Loss
l1, mse, bce = nn.L1Loss(), nn.MSELoss(), nn.BCELoss()

# Initialize Optimizer
G_optimizer = optim.Adam(
    [{"params": decoder.parameters()}, {"params": encoder.parameters()}],
    lr=lr_rate,
    betas=(0.5, 0.999),
)

D_optimizer = optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(0.5, 0.999))
scheduler = StepLR(G_optimizer, step_size=epochs_per_stage, gamma=0.9)

min_pred_error = np.inf
for epoch in range(num_epochs):
    D_loss_sum, G_loss_sum, pair_div_loss_sum = 0, 0, 0

    for i, inputs in enumerate(loader):
        ########## Inputs ########
        images, states, actions = inputs

        images, states, actions = (
            images.float().to(gpu_id),
            states.float().to(gpu_id),
            actions.float().to(gpu_id),
        )

        ######## Unpack trajectories ########
        state_cur, state_target = torch.split(
            images, split_size_or_sections=[dataset.seq_length - 1, 1], dim=1
        )

        state_cur = state_cur.reshape(-1, *(state_cur.size()[2:]))

        state_target = torch.repeat_interleave(
            state_target.squeeze(dim=1), repeats=dataset.seq_length - 1, dim=0
        )

        actions = actions[:, :-1].reshape(-1, actions.size()[-1])

        ######## Ground truth copies repeated by number samples ########
        action_unsqueeze = torch.repeat_interleave(actions, repeats=num_sample, dim=0)

        state_cur_unsqueeze = torch.repeat_interleave(
            state_cur, repeats=num_sample, dim=0
        )
        state_target_unsqueeze = torch.repeat_interleave(
            state_target, repeats=num_sample, dim=0
        )

        ########## Encode Images ########
        state_codes = encoder(state_cur).detach()
        target_codes = encoder(state_target).detach()

        codes = torch.cat([state_codes, target_codes], dim=1).squeeze()
        codes_unsqueeze = torch.repeat_interleave(codes, repeats=num_sample, dim=0)

        ########## diverse noise sampling ########
        diverse_codes, noises = diverse_sampling(codes)
        diverse_codes, noises = diverse_codes[..., None, None], noises[..., None, None]

        action_hat = decoder(diverse_codes.view(-1, diverse_codes.size(2)))

        ################## USEFUL CONSTANTS ##################
        FLAT_BATCH_SIZE = batch_size * (dataset.seq_length - 1)
        DIV_BATCH_SIZE = num_sample * FLAT_BATCH_SIZE

        ################## Train Discriminator ##################
        for _ in range(discr_steps_per_gen):

            D_loss = nn.BCEWithLogitsLoss()(
                torch.squeeze(discriminator(action_unsqueeze, codes_unsqueeze)),
                torch.ones(DIV_BATCH_SIZE).to(gpu_id),
            ) + nn.BCEWithLogitsLoss()(
                torch.squeeze(discriminator(action_hat, codes_unsqueeze)),
                torch.zeros(DIV_BATCH_SIZE).to(gpu_id),
            )

            D_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            D_optimizer.step()

        ########## G Loss ##########
        G_loss = nn.BCEWithLogitsLoss()(
            torch.squeeze(discriminator(action_hat, codes_unsqueeze)),
            torch.ones(DIV_BATCH_SIZE).to(gpu_id),
        )

        ########## Div Loss ##########
        pair_div_loss = div.compute_pairwise_divergence(
            action_hat.view(FLAT_BATCH_SIZE, num_sample, -1),
            noises.squeeze(3).squeeze(3),
        )

        total_loss = G_loss + 0.1 * pair_div_loss
        G_optimizer.zero_grad()
        total_loss.backward()
        G_optimizer.step()

        D_loss_sum += D_loss.cpu().data.numpy()
        G_loss_sum += G_loss.cpu().data.numpy()
        pair_div_loss_sum += pair_div_loss.cpu().data.numpy()

    D_loss_avg = D_loss_sum / len(loader)
    G_loss_avg = G_loss_sum / len(loader)
    pair_div_loss_avg = pair_div_loss_sum / len(loader)

    print(
        epoch, "D: ", D_loss_avg, "G: ", G_loss_avg, "div: ", pair_div_loss_avg,
    )

    display.plot("gan", "discriminator", "GAN Loss", epoch, D_loss_avg)
    display.plot("gan", "generator", "GAN Loss", epoch, G_loss_avg)
    display.plot(
        "pairwise_div", "loss", "Pairwise Divergence Loss", epoch, pair_div_loss_avg
    )

    if epoch % epochs_per_stage == epochs_per_stage - 1:

        if not os.path.exists("models/gan"):
            os.makedirs("models/gan")

        torch.save(discriminator, "models/gan/gan_discriminator_" + str(epoch) + ".pt")
        torch.save(decoder, "models/gan/gan_decoder_" + str(epoch) + ".pt")

    # if step % report_feq == 0:

    #     state_cur_vis = denorm(state_cur[0]).detach().cpu().numpy().astype(np.uint8)
    #     action_gt = action[0].detach().cpu().numpy()
    #     action_hat_1 = action_hat[0].detach().cpu().numpy()
    #     action_hat_2 = action_hat[1].detach().cpu().numpy()
    #     action_hat_3 = action_hat[2].detach().cpu().numpy()

    #     gt_color = (255,0,0)
    #     state_cur_arrow_gt = draw_action_arrow(state_cur_vis, action_gt, gt_color)

    #     hat_color = (0,0,255)
    #     state_cur_arrow_hat_1 = draw_action_arrow(state_cur_vis, action_hat_1, hat_color)
    #     state_cur_arrow_hat_2 = draw_action_arrow(state_cur_vis, action_hat_2, hat_color)
    #     state_cur_arrow_hat_3 = draw_action_arrow(state_cur_vis, action_hat_3, hat_color)

    #     display.img_result(state_cur_arrow_gt, win=1, caption="state_cur_arrow_gt")
    #     display.img_result(state_cur_arrow_hat_1, win=2, caption="state_cur_arrow_hat_1")
    #     display.img_result(state_cur_arrow_hat_2, win=3, caption="state_cur_arrow_hat_2")
    #     display.img_result(state_cur_arrow_hat_3, win=4, caption="state_cur_arrow_hat_3")
