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

## View individual trajectory

To view an individual trajectory:
```
python -m utils.trajectory_loader _path_to_data_
```