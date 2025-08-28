# Foundation Policies with Hilbert Representations

## Overview
This is the official implementation of **[Hilbert Foundation Policies](https://seohong.me/projects/hilp/)** (**HILPs**).

This repository contains two different implementations of HILPs:
* For the zero-shot RL experiments, please refer to the [hilp_zsrl](hilp_zsrl) directory.
* For the goal-conditioned RL experiments, please refer to the [hilp_gcrl](hilp_gcrl) directory.

Visit [our project page](https://seohong.me/projects/hilp/) for more details.

## License
MIT

## Hilbert Representations
```
CUDA_VISIBLE_DEVICES=0 python /root/workspace/HILP/train_hilbert_representation.py --data_dir /root/workspace/HugWBC/collected_trajectories_v2 --output_dir /root/workspace/HILP/logs/hilbert_test_output_proprio --epochs 1 --batch_size 512 --types constant switch --max_episodes_per_type 1 --use_layer_norm --representation_dim 32 --learning_rate 1e-4 --gamma 0.98 --expectile 0.5 --tau 0.005 --device cuda

CUDA_VISIBLE_DEVICES=0 python /root/workspace/HILP/train_hilbert_representation.py \
  --data_dir /root/workspace/HugWBC/collected_trajectories_v2 \
  --output_dir /root/workspace/HILP/logs/representation_5 \
  --epochs 100 \
  --batch_size 2048 \
  --types constant switch \
  --representation_dim 32 \
  --learning_rate 1e-4 \
  --gamma 0.99 \
  --expectile 0.5 \
  --tau 0.005 \
  --obs_horizon 5 \
  --device cuda \
  --wandb --wandb_project hilbert_training --wandb_group zsrl_proprio --wandb_mode online

CUDA_VISIBLE_DEVICES=3 python /root/workspace/HILP/train_hilbert_representation.py \
  --data_dir /root/workspace/HugWBC/collected_trajectories_v2 \
  --output_dir /root/workspace/HILP/logs/representation_5_1 \
  --epochs 100 \
  --batch_size 2048 \
  --types constant switch \
  --representation_dim 32 \
  --learning_rate 1e-4 \
  --gamma 0.99 \
  --goal_future 0.99 \
  --p_trajgoal 0.9  \
  --p_randomgoal 0.1 \
  --expectile 0.5 \
  --tau 0.005 \
  --obs_horizon 5 \
  --device cuda \
  --wandb --wandb_project hilbert_training --wandb_group zsrl_proprio --wandb_mode online

CUDA_VISIBLE_DEVICES=1 python /root/workspace/HILP/train_hilbert_representation.py \
  --data_dir /root/workspace/HugWBC/collected_trajectories_v2 \
  --output_dir /root/workspace/HILP/logs/representation_3 \
  --epochs 100 \
  --batch_size 2048 \
  --types constant switch \
  --representation_dim 32 \
  --learning_rate 1e-4 \
  --gamma 0.99 \
  --expectile 0.5 \
  --tau 0.005 \
  --obs_horizon 3 \
  --device cuda \
  --wandb --wandb_project hilbert_training --wandb_group zsrl_proprio --wandb_mode online

CUDA_VISIBLE_DEVICES=2 python /root/workspace/HILP/train_hilbert_representation.py \
  --data_dir /root/workspace/HugWBC/collected_trajectories_v2 \
  --output_dir /root/workspace/HILP/logs/representation_1 \
  --epochs 100 \
  --batch_size 2048 \
  --types constant switch \
  --representation_dim 32 \
  --learning_rate 1e-4 \
  --gamma 0.99 \
  --expectile 0.5 \
  --tau 0.005 \
  --obs_horizon 1 \
  --device cuda \
  --wandb --wandb_project hilbert_training --wandb_group zsrl_proprio --wandb_mode online
```

```
CUDA_VISIBLE_DEVICES=7 python /root/workspace/HILP/train_hilp_feature.py \
  --data_dir /root/workspace/HugWBC/collected_trajectories_v2 \
  --output_dir /root/workspace/HILP/logs/hilp_feature_runs_contant_only \
  --types constant \
  --obs_horizon 5 --goal_future 0.98 --p_trajgoal 0.8 \
  --p_randomgoal 0.2  \
  --epochs 100 --batch_size 2048 --learning_rate 1e-4 --weight_decay 1e-4 \
  --gamma 0.98 --expectile 0.5 --device cuda \
  --wandb --wandb_project hilbert_feature --wandb_group zsrl_hilp

CUDA_VISIBLE_DEVICES=2 python /root/workspace/HILP/train_hilbert_representation.py \
  --data_dir /root/workspace/HugWBC/collected_trajectories_v2 \
  --output_dir /root/workspace/HILP/logs/representation_5 \
  --epochs 100 \
  --batch_size 2048 \
  --types constant switch \
  --representation_dim 32 \
  --learning_rate 1e-4 \
  --gamma 0.99 \
  --expectile 0.5 \
  --tau 0.005 \
  --obs_horizon 5 --goal_future 0.98 --p_trajgoal 0.8 \
  --p_randomgoal 0.2  \
  --device cuda \
  --wandb --wandb_project hilbert_training --wandb_group zsrl_proprio --wandb_mode online
```