0923
```
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_online.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run agent.goal_type=state goal_eval=True agent.obs_horizon=1

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_online_fused.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run agent.goal_type=state goal_eval=True agent.obs_horizon=1
```

```
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline_rds.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd agent.goal_type=state goal_eval=True agent.obs_horizon=1 

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline_rds.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd agent.goal_type=state goal_eval=True agent.obs_horizon=1 

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline_rds_t.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd phi_total_episodes=5000 agent.goal_type=state goal_eval=True agent.obs_horizon=1 phi_rollout_num=50

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONPATH=. python url_benchmark/train_offline_rds_t.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd phi_total_episodes=5000 agent.goal_type=state goal_eval=True agent.obs_horizon=1 local_replay_buffer_path=/root/workspace/exorl/datasets/walker/rnd/replay.pt

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 PYTHONPATH=. python -m debugpy --listen 5678 --wait-for-client url_benchmark/train_offline_rds_t.py run_group=WALKER device=cuda agent=sf_v2_online agent.feature_learner=hilp p_randomgoal=0.375 agent.hilp_expectile=0.5 agent.hilp_discount=0.96 agent.q_loss=False seed=0 task=walker_run expl_agent=rnd phi_total_episodes=5000 agent.goal_type=state goal_eval=True agent.obs_horizon=1  local_replay_buffer_path=/root/workspace/exorl/datasets/walker/rnd/replay.pt
```