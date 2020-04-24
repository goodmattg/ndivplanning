Repository for curiosity/generative planning/3d research

Credit to Lingzhi and Andong Cao for sharing their code.

```
@misc{zhang2019neural,
    title={Neural Embedding for Physical Manipulations},
    author={Lingzhi Zhang and Andong Cao and Rui Li and Jianbo Shi},
    year={2019},
    eprint={1907.06143},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Credit to Tianhong Dai for his implementation of HER in OpenAI Gym robotics environments.
https://github.com/TianhongDai/hindsight-experience-replay

# Getting Started

# Pretrained HER

Download Tianhong Dai's pretrained HER models from:
https://drive.google.com/open?id=18MEEwweR8Ad1t1yxgTwI1VYmIsVPttXk

```
mkdir models/her_pretrained/FetchPush-v1
mv model.pt models/her_pretrained/FetchPush-v1
```

## Generating trajectories

**Argument documentation found with `python generate_trajectories.py -h`**

To generate trajectories with default arguments (1 file, 1000 trajectories for file, saved to /data):
```
python generate_trajectories.py
```

Semi-random forward goal trajectories. Object is placed, goal then placed in pi/3 radius in front of object.
```
python generate_trajectories.py --trajectory-length 8 --num_trajectory_per_file 1000 --filename_start_idx 1 --num_files 20 --outdir data/forward_inline/train --image-shape 128 128 --simplify-task
```

Inline goal trajectories:
```
python generate_trajectories.py --trajectory-length 8 --num_trajectory_per_file 1000 --filename_start_idx 1 --num_files 20 --outdir data/forward_inline/train --image-shape 128 128 --simplify-task --goal-inline
```

## View individual trajectory

To view an individual trajectory:
```
python -m utils.trajectory_loader _path_to_data_
```

## Training the forward model
```
python train_forward_model.py --trajectory-length 8 --train-data-path forward_inline_data/train --gpu-id 2 --log-port 8081 --forward-save-path models/inline_task_forward
```

## Training the GAN
python train_gan.py --trajectory-length 8 --train-data-path forward_inline_data/train --gpu-id 2 --log-port 8081 --gan-save-path models/inline_task_gan