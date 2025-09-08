```
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/hilp_zsrl/bin/python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py \
  load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 \
  use_wandb=True \
  agent=sf agent.batch_size=1024 \
  checkpoint_every=0 \
  hilbert_obs_horizon=5 run_final_eval=False

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/hilp_zsrl/bin/python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py \
  load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 \
  use_wandb=True \
  agent=sf agent.batch_size=1024 \
  num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=0 \
  hilbert_obs_horizon=5 \
  task=h1int

CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=False agent=sf agent.batch_size=64 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000  hilbert_obs_horizon=5 task=h1int 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000  hilbert_obs_horizon=5 task=h1int

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python  -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000  hilbert_obs_horizon=5 task=h1int


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False

CUDA_VISIBLE_DEVICES=3 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -m debugpy --listen 5678 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False
```

command injection, no random goal no mix
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:0 use_history_action=False

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:6 p_randomgoal=0.375
```

train phi only for test
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_phi.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:7

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_phi.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.98 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:5
```

# 0901 2039
```
# command injection
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:0 use_history_action=False
# no command injection
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:1 use_history_action=False
# large z dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:1 use_history_action=False agent.z_dim=128
# command injection failed, the generated z conditioning on actual command does not align with the hilbert space
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_trajectories_v2 use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0 device=cuda:3 use_history_action=False agent.z_dim=128
# small traj length eval only
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:3 use_history_action=False agent.z_dim=128 eval_only=True load_model=/root/workspace/HILP/hilp_zsrl/exp_local/Debug/sd1...20250902072734.Eh1int.Msf_offline/models

# small traj length with random goal
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:2 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True

# small traj length with random goal, large phi and large z_dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:3 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048

# small traj length with random goal, orig scripts for compare
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True

# small traj length with random goal, large phi and large z_dim, orig scripts for compare
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:3 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048

# small traj length with no random goal
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:2 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True

# small traj length with no random goal, large phi and large z_dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048

HYDRA_FULL_ERROR=1 CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:2 p_randomgoal=0.375 use_history_action=False agent.z_dim=128 agent.use_large_phi_net=True

# small traj length with random goal and critic obs with disentangled
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:2 p_randomgoal=0.375 use_history_action=False agent.z_dim=128 agent.use_large_phi_net=True
```

```dl and origin scritps comparison
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:7 use_history_action=False agent.z_dim=128

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0 device=cuda:6 use_history_action=False agent.z_dim=128

```

# exp groups
```
# random goal, large net, large dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 
# no random goal, large net, large dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048

# random goal, large net, small dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:2 p_randomgoal=0.375 use_history_action=False agent.z_dim=256 agent.use_large_phi_net=True agent.phi_hidden_dim=2048
# no random goal, large net, small dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:3 p_randomgoal=0 use_history_action=False agent.z_dim=256 agent.use_large_phi_net=True agent.phi_hidden_dim=2048
```
# for eval
```
# random goal, large net, large dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 eval_only=True  load_model=/root/workspace/HILP/hilp_zsrl/exp_local/sf_h1int_0.98_f0.99_pr0.375_phi_exp0.5_phi_g0.96_qlFalse_False_mix0.5_False_512_collected_single_short_phih2048/20250904072510/models

# no random goal, large net, large dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 eval_only=True load_model=/root/workspace/HILP/hilp_zsrl/exp_local/sf_h1int_0.98_f0.99_pr0.0_phi_exp0.5_phi_g0.96_qlFalse_False_mix0.5_False_512_collected_single_short_phih2048/20250904072510/models

# random goal, large net, large dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python -m debugpy --listen 5678 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=False agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

# no random goal, large net, large dim
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:6 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048  agent.feature_type=diff
```

```
# full training with phi only, with random goal
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_phi.py load_replay_buffer=/root/workspace/HugWBC/dataset/Mixture use_wandb=True agent=sf agent.batch_size=2048 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:5 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048  agent.feature_type=diff
```

# debug
```
HYDRA_FULL_ERROR=1 python -m debugpy --listen 5678 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=False agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff
HYDRA_FULL_ERROR=1 python -m debugpy --listen 5678 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=False agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state
```

# rerun
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:7 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:6 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state
```

```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff resume_from=/root/workspace/HILP/hilp_zsrl/exp_local/sf_h1int_0.98_f0.99_pr0.375_phi_exp0.5_phi_g0.96_qlFalse_False_mix0.5_False_512_collected_single_short_phih2048_diff/20250908090431

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state resume_from=/root/workspace/HILP/hilp_zsrl/exp_local/sf_h1int_0.98_f0.99_pr0.375_phi_exp0.5_phi_g0.96_qlFalse_False_mix0.5_False_512_collected_single_short_phih2048_state/20250908090435

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:7 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff resume_from=/root/workspace/HILP/hilp_zsrl/exp_local/sf_h1int_0.98_f0.99_pr0.0_phi_exp0.5_phi_g0.96_qlFalse_False_mix0.5_False_512_collected_single_short_phih2048_diff/20250908090105

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:6 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state resume_from=/root/workspace/HILP/hilp_zsrl/exp_local/sf_h1int_0.98_f0.99_pr0.0_phi_exp0.5_phi_g0.96_qlFalse_False_mix0.5_False_512_collected_single_short_phih2048_state/20250908090112
```

# full training
```
HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/Mixture use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:5 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short_new use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short_new use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short_new use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0.5 device=cuda:2 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short_new use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=True agent.mix_ratio=0.5 device=cuda:3 p_randomgoal=0.375 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short_new use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:0 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=diff

HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged_dl.py load_replay_buffer=/root/workspace/HugWBC/dataset/collected_single_short_new use_wandb=True agent=sf agent.batch_size=1024 num_grad_steps=100000000 eval_every_steps=10000 hilbert_obs_horizon=5 task=h1int agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False agent.command_injection=False agent.mix_ratio=0.5 device=cuda:1 p_randomgoal=0 use_history_action=False agent.z_dim=512 agent.use_large_phi_net=True agent.phi_hidden_dim=2048 agent.feature_type=state
```