import argparse
import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import h5py
import io

from PIL import Image
from dotmap import DotMap

from utils.argparse_util import *
from hindsight_experience_replay.rl_modules.models import actor

# This is copied from hindsight_experience_replay demo.py
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


def generate_trajectory(env, actor_network, args):
    """    
    Inputs:
        env: OpenAI environment object
        actor_network: network to generate actions

    Outputs:
        image_frames: np.ndarray(seq_length,) ByteArray
        states: np.ndarray(seq_length, 25)
        actions: np.ndarray(seq_length, 4)

    """
    observation = env.reset()

    # If we want a simpler task, place the object close to the gripper initial position with goal 'in front of' object
    if args.simplify_task:

        object_xpos = env.initial_gripper_xpos[:2].copy()

        # Sample a random direction
        theta = np.random.rand() * (2 * np.pi)
        offset = np.array([np.cos(theta), np.sin(theta)])

        # Account for object width
        object_xpos = object_xpos + offset * 0.056

        if args.goal_inline:
            # Goal should explicitly be inline with object
            env.env.goal = np.hstack(
                [
                    object_xpos.copy() + offset * np.random.uniform(0.1, 0.11),
                    env.height_offset,
                ]
            )
        else:
            # Goal should be in front of object
            goal_theta = np.random.uniform(theta - np.pi / 3, theta + np.pi / 3)

            goal_offset = np.array(
                [np.cos(goal_theta), np.sin(goal_theta)]
            ) * np.random.uniform(0.16, 0.18)

            env.env.goal = np.hstack(
                [env.initial_gripper_xpos[:2].copy() + goal_offset, env.height_offset]
            )

        object_qpos = env.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        env.env.sim.data.set_joint_qpos("object0:joint", object_qpos)

        env.sim.forward()

        # New observation with updated state
        observation = env.env._get_obs()

    # start to do the demo
    obs = observation["observation"]
    g = observation["desired_goal"]

    image_frames = []
    states = np.empty((args.trajectory_length, 25))
    actions = np.empty((args.trajectory_length, 4))

    for ix in range(args.trajectory_length):

        # Store the frame as jpeg encoded byte array
        image = render(env)
        im_byte_arr = io.BytesIO()
        im = Image.fromarray(image)

        im = im.resize(args.image_shape, Image.LANCZOS)

        im.save(im_byte_arr, format="jpeg", quality=95)

        image_frames.append(im_byte_arr.getvalue())

        inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
        with torch.no_grad():
            pi = actor_network(inputs)
        action = pi.detach().numpy().squeeze()

        states[ix] = obs  # Store the state
        actions[ix] = action  # Store the action

        # put actions into the environment
        observation_new, reward, _, info = env.step(action)
        obs = observation_new["observation"]

    return image_frames, states, actions, observation["desired_goal"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-name", type=str, default="FetchPush-v1", help="the environment name"
    )

    parser.add_argument(
        "--trajectory-length", type=int, default=20, help="the demo length"
    )

    parser.add_argument("--simplify-task", dest="simplify_task", action="store_true")

    parser.add_argument("--goal-inline", dest="goal_inline", action="store_true")

    parser.add_argument(
        "--pretrained_model_path",
        type=file_exists,
        default="models/her_pretrained/FetchPush-v1/model.pt",
    )

    parser.add_argument("--clip-obs", type=float, default=200, help="the clip ratio")

    parser.add_argument("--clip-range", type=float, default=5, help="the clip range")

    parser.add_argument(
        "--num_trajectory_per_file",
        type=int,
        default=1000,
        help="number of trajectories to include in each file",
    )

    parser.add_argument(
        "--num_files",
        type=int,
        default=1,
        help="each file will contain num_trajectory_per_file trajectories",
    )

    parser.add_argument(
        "--filename_start_idx",
        type=int,
        default=1,
        help="Plain-text label to start filename. i.e. default is file_0001",
    )

    parser.add_argument(
        "--image-shape",
        nargs=2,
        type=int,
        default=(500, 500),
        help="Output image shape (WIDTH, HEIGHT) via PIL.Image.resize()",
    )

    parser.add_argument(
        "--outdir",
        default="data",
        type=dir_exists_write_privileges,
        help="Data file storage directory path",
    )

    parser.set_defaults(simplify_task=False, goal_inline=False)

    args = parser.parse_args()

    # OpenAI gym render() returns (500, 500, 3) image by default
    ORIGINAL_WIDTH = 500
    ORIGINAL_HEIGHT = 500
    COLOR_CHANNELS = 3
    # OpenAI gym FetchReach-v1 state dim. is 25
    # See line 112 for composition details:
    # https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py
    STATE_DIM = 25
    # OpenAI gym FetchReach-v1 action dim. is 4
    ACTION_DIM = 4
    SEQ_LENGTH = args.trajectory_length

    ######################################################################
    # Create and configure the environment
    ######################################################################
    env = gym.make(args.env_name)

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

    # NOTE: The OpenAI environment throws an error if this isn't here
    _ = render(env)

    env.viewer.cam.distance = 1.0
    env.viewer.cam.azimuth = 130
    env.viewer.cam.elevation = -40.0

    ######################################################################
    # Create the HER actor network
    ######################################################################
    o_mean, o_std, g_mean, g_std, model = torch.load(
        args.pretrained_model_path, map_location=lambda storage, loc: storage
    )

    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()

    for fidx in range(args.num_files):

        out_file = os.path.join(
            args.outdir,
            "trajectory_bundle_{:05d}.h5".format(args.filename_start_idx + fidx),
        )

        with h5py.File(out_file, "w") as f:

            for ix in range(args.num_trajectory_per_file):

                try:

                    ####################################
                    # Generate the trajectory
                    ####################################
                    image_frames, states, actions, goal = generate_trajectory(
                        env, actor_network, args
                    )

                    ####################################
                    # Save the trajectory
                    ####################################
                    file_group = f.create_group("trajectory_{:05d}".format(ix))

                    img_ds = file_group.create_dataset(
                        "images",
                        (SEQ_LENGTH,),
                        data=image_frames,
                        compression="gzip",
                        compression_opts=9,
                    )

                    img_ds.attrs["description"] = np.string_("raw_pixels")
                    img_ds.attrs["shape"] = np.array(
                        [SEQ_LENGTH, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHANNELS],
                        dtype="int32",
                    )

                    state_ds = file_group.create_dataset(
                        "states",
                        (SEQ_LENGTH, STATE_DIM),
                        dtype="float32",
                        data=states,
                        compression="gzip",
                        compression_opts=9,
                    )

                    state_ds.attrs["description"] = np.string_(
                        "gripper_and_object_position_velocity_rotation"
                    )
                    state_ds.attrs["shape"] = np.array(states.shape, dtype="int32")

                    action_ds = file_group.create_dataset(
                        "actions",
                        (SEQ_LENGTH, ACTION_DIM),
                        dtype="float32",
                        data=actions,
                        compression="gzip",
                        compression_opts=9,
                    )

                    action_ds.attrs["description"] = np.string_("action_tensor")
                    action_ds.attrs["shape"] = np.array(actions.shape, dtype="int32")

                    goal_ds = file_group.create_dataset(
                        "goal",
                        (3,),
                        dtype="float32",
                        data=goal,
                        compression="gzip",
                        compression_opts=9,
                    )

                except Exception as e:
                    print(e)
                    print("Unable to create trajectory: {:05d}".format(ix))
