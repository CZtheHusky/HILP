# HILP Online
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_hum_online.py  use_wandb=True agent=sf_hum_online agent.batch_size=1024 agent.obs_horizon=1 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0 use_history_action=False agent.z_dim=64 agent.phi_hidden_dim=1024 agent.feature_type=diff agent.random_sample_z=False
```

# Foundation Policies with Hilbert Representations (Zero-Shot RL)

## Overview
This codebase contains the official implementation of **[Hilbert Foundation Policies](https://seohong.me/projects/hilp/)** (**HILPs**) for **zero-shot RL**.
The implementation is based on [Does Zero-Shot Reinforcement Learning Exist?](https://github.com/facebookresearch/controllable_agent).

## Requirements
* Python 3.8

## Installation
```
conda create --name=hilp_zsrl python=3.8 -y && conda activate hilp_zsrl
pip install -r requirements.txt
```

## Examples
```
CUDA_VISIBLE_DEVICES=6 python train_hilp_feature.py  --data_dir /root/workspace/HugWBC/dataset/collected_trajectories_v2  --discount 0.98   --future 0.99   --p_randomgoal 0.375 --epochs 100 --batch_size 2048   --learning_rate 5e-4   --device auto --wandb --z_dim 50

CUDA_VISIBLE_DEVICES=5 python train_hilp_feature.py  --data_dir /root/workspace/HugWBC/dataset/collected_trajectories_v2  --discount 0.98   --future 0.99   --p_randomgoal 0.375 --epochs 100 --batch_size 2048   --learning_rate 5e-4   --device auto --wandb --z_dim 50 --gamma 0.965

CUDA_VISIBLE_DEVICES=4 python train_hilp_feature.py  --data_dir /root/workspace/HugWBC/dataset/collected_trajectories_v2  --discount 0.98   --future 0.99   --p_randomgoal 0.375 --epochs 100 --batch_size 2048   --learning_rate 5e-4   --device auto --wandb --z_dim 50 --gamma 0.97

CUDA_VISIBLE_DEVICES=7 python train_hilp_feature.py  --load_replay_buffer /root/workspace/exorl/datasets/walker/rnd/replay.pt --discount 0.98   --future 0.99   --p_randomgoal 0.375 --epochs 100 --batch_size 2048   --learning_rate 5e-4   --device auto --wandb --z_dim 50

```

### State-Based ExORL
* Download the ExORL datasets following the instructions at https://github.com/denisyarats/exorl.
* Convert the dataset using `convert.py`:
```
# Covert RND Walker 
python convert.py --save_path=PATH_TO_SAVE --env=walker --task=run --method=rnd --num_episodes=5000 --use_pixels=0
# Convert RND Cheetah 
python convert.py --save_path=PATH_TO_SAVE --env=cheetah --task=run --method=rnd --num_episodes=5000 --use_pixels=0
# Convert RND Quadruped
python convert.py --save_path=PATH_TO_SAVE --env=quadruped --task=run --method=rnd --num_episodes=5000 --use_pixels=0
# Convert RND Jaco
python convert.py --save_path=PATH_TO_SAVE --env=jaco --task=reach_top_left --method=rnd --num_episodes=20000 --use_pixels=0
```
* Train policies:
```
# HILP on RND Walker
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000

PYTHONPATH=. python -m debugpy --listen 5678 --wait-for-client  url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000

PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf_v agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.feature_type=diff

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf_v2 agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.goal_type=delta

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf_v2 agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.goal_type=state

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 PYTHONPATH=. python -m debugpy --listen 5678 --wait-for-client url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf_v agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.feature_type=diff

PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=True seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000

PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.mix_ratio=0

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000

PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.z_dim=24

PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/root/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000 agent.feature_type=diff goal_eval=True

PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=/cpfs/user/caozhe/workspace/exorl/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000
# HILP on RND Cheetah
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=False seed=0 task=cheetah_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/cheetah/rnd/replay.pt replay_buffer_episodes=5000
# HILP on RND Quadruped
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=True seed=0 task=quadruped_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/quadruped/rnd/replay.pt replay_buffer_episodes=5000
# HILP on RND Jaco
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=True seed=0 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/jaco/rnd/replay.pt replay_buffer_episodes=20000

# HILP-G on RND Jaco
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.9 agent.feature_type=diff goal_eval=True agent.hilp_discount=0.98 agent.q_loss=True seed=0 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/jaco/rnd/replay.pt replay_buffer_episodes=20000

# FB on RND Walker
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=fb_ddpg agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000
# FB on RND Cheetah
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=fb_ddpg agent.q_loss=False seed=0 task=cheetah_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/cheetah/rnd/replay.pt replay_buffer_episodes=5000
# FB on RND Quadruped
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=fb_ddpg agent.q_loss=False seed=0 task=quadruped_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/quadruped/rnd/replay.pt replay_buffer_episodes=5000
# FB on RND Jaco
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=fb_ddpg agent.q_loss=False seed=0 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/jaco/rnd/replay.pt replay_buffer_episodes=20000

# FDM on RND Walker
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=transition agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/walker/rnd/replay.pt replay_buffer_episodes=5000
# FDM on RND Cheetah
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=transition agent.q_loss=False seed=0 task=cheetah_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/cheetah/rnd/replay.pt replay_buffer_episodes=5000
# FDM on RND Quadruped
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=transition agent.q_loss=True seed=0 task=quadruped_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/quadruped/rnd/replay.pt replay_buffer_episodes=5000
# FDM on RND Jaco
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=transition agent.q_loss=True seed=0 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/jaco/rnd/replay.pt replay_buffer_episodes=20000
```

### Pixel-Based ExORL
* Download the ExORL datasets following the instructions at https://github.com/denisyarats/exorl.
* Convert the dataset using `convert.py`:
```
# Convert RND Walker
python convert.py --save_path=PATH_TO_SAVE --env=walker --task=run --method=rnd --num_episodes=5000 --use_pixels=1
# Convert RND Cheetah
python convert.py --save_path=PATH_TO_SAVE --env=cheetah --task=run --method=rnd --num_episodes=5000 --use_pixels=1
# Convert RND Quadruped
python convert.py --save_path=PATH_TO_SAVE --env=quadruped --task=run --method=rnd --num_episodes=5000 --use_pixels=1
# Convert RND Jaco
python convert.py --save_path=PATH_TO_SAVE --env=jaco --task=reach_top_left --method=rnd --num_episodes=20000 --use_pixels=1
```
* Train policies:
```
# HILP on pixel-based RND Walker
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/walker/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
# HILP on pixel-based RND Cheetah
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=False seed=0 task=cheetah_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/cheetah/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
# HILP on pixel-based RND Quadruped
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=True seed=0 task=quadruped_run expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/quadruped/rnd/replay_pixel64.pt replay_buffer_episodes=5000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
# HILP on pixel-based RND Jaco
PYTHONPATH=. python url_benchmark/train_offline.py run_group=EXP device=cuda agent=sf agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=True seed=0 task=jaco_reach_top_left expl_agent=rnd load_replay_buffer=PATH_TO_DATASET/datasets/jaco/rnd/replay_pixel64.pt replay_buffer_episodes=20000 obs_type=pixels agent.batch_size=512 num_grad_steps=500000
```

## License
MIT