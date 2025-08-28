```
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/hilp_zsrl/bin/python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py \
  load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 \
  use_wandb=True save_video=False device=cuda \
  agent=sf agent.batch_size=1024 \
  checkpoint_every=0 \
  legged_use_actions=True legged_use_rewards=True \
  hilbert_obs_horizon=5 run_final_eval=False

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/hilp_zsrl/bin/python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py \
  load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 \
  use_wandb=True save_video=False device=cuda \
  agent=sf agent.batch_size=1024 \
  num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=0 \
  legged_use_actions=True legged_use_rewards=True \
  hilbert_obs_horizon=5 run_final_eval=False \
  task=h1int

CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=False save_video=False device=cuda agent=sf agent.batch_size=64 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000 legged_use_actions=True legged_use_rewards=True hilbert_obs_horizon=5 run_final_eval=False task=h1int 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True save_video=False device=cuda agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000 legged_use_actions=True legged_use_rewards=True hilbert_obs_horizon=5 run_final_eval=False task=h1int

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python  -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True save_video=False device=cuda agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000 legged_use_actions=True legged_use_rewards=True hilbert_obs_horizon=5 run_final_eval=False task=h1int


CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python  -m debugpy --listen 5679 --wait-for-client /root/workspace/HILP/hilp_zsrl/url_benchmark/train_offline_legged.py load_replay_buffer=/root/workspace/HugWBC/collected_trajectories_v2 use_wandb=True save_video=False device=cuda agent=sf agent.batch_size=1024 num_grad_steps=1000000 eval_every_steps=1000000 checkpoint_every=1000 legged_use_actions=True legged_use_rewards=True hilbert_obs_horizon=5 run_final_eval=False task=h1int
```