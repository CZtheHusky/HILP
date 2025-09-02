```
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/hilp_zsrl/bin/python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py \
  load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 \
  use_wandb=True \
  agent=sf agent.batch_size=1024 \
  checkpoint_every=0 \
  hilbert_obs_horizon=5 run_final_eval=False

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/hilp_zsrl/bin/python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py \
  load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 \
  use_wandb=True \
  agent=sf agent.batch_size=1024 \
  num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=0 \
  hilbert_obs_horizon=5 \
  task=h1int

CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=False agent=sf agent.batch_size=64 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000  hilbert_obs_horizon=5 task=h1int 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000  hilbert_obs_horizon=5 task=h1int

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python  -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000  hilbert_obs_horizon=5 task=h1int


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -m debugpy --listen 5678 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False
```

command injection, no random goal no mix
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:0 use_history_action=False

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:6 p_randomgoal=0.375
```

train phi only for test
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_phi.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:7

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_phi.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:5
```

# 0901 2039
```
# command injection
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:0 use_history_action=False
# no command injection
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:1 use_history_action=False
# large z dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:2 use_history_action=False agent.z_dim=128
# command injection failed, the generated z conditioning on actual command does not align with the hilbert space
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:3 use_history_action=False agent.z_dim=128
# small traj length
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 checkpoint_every=10000  hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:3 use_history_action=False agent.z_dim=128 
```