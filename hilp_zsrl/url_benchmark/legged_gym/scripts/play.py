import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from isaacgym import gymapi
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi

def play(args):
    # 获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)
    
    # ==================== 环境参数覆盖设置 ====================
    # 覆盖训练时的环境数量，测试时只使用1个环境，便于观察和调试
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    # 设置很长的episode长度，避免自动重置，便于长时间观察
    env_cfg.env.episode_length_s = 100000

    # ==================== 地形设置 ====================
    # 关闭地形课程学习，保持固定难度
    env_cfg.terrain.curriculum = False
    # 启用噪声添加
    env_cfg.noise.add_noise = True
    
    # ==================== 域随机化设置 ====================
    # 启用摩擦系数随机化，增加测试的真实性
    env_cfg.domain_rand.randomize_friction = True
    # 关闭负载随机化，保持一致性
    env_cfg.domain_rand.randomize_load = False
    # 关闭增益随机化，保持一致性
    env_cfg.domain_rand.randomize_gains = False 
    # 关闭链接属性随机化，保持一致性
    env_cfg.domain_rand.randomize_link_props = False
    # 关闭基座质量随机化，保持一致性
    env_cfg.domain_rand.randomize_base_mass = False

    # ==================== 命令和奖励设置 ====================
    # 设置命令重采样时间（100秒），减少命令变化频率
    env_cfg.commands.resampling_time = 100
    # 关闭奖励课程学习，保持固定奖励结构
    env_cfg.rewards.penalize_curriculum = False
    
    # ==================== 地形类型和参数设置 ====================
    # 设置地形类型为三角网格
    env_cfg.terrain.mesh_type = 'trimesh'
    # 简化为单行单列地形
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    # 设置初始地形难度等级
    env_cfg.terrain.max_init_terrain_level = 1
    # 选择特定地形类型而不是随机生成
    env_cfg.terrain.selected = True
    # 选择均匀随机地形类型
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    # 地形参数配置：创建平地地形
    env_cfg.terrain.terrain_kwargs = {  # Dict of arguments for selected terrain
        "random_uniform":
            {
                "min_height": -0.00,      # 最小高度
                "max_height": 0.00,       # 最大高度（平地）
                "step": 0.005,            # 高度步长
                "downsampled_scale": 0.2  # 下采样比例
            },
    }

    # ==================== 环境创建 ====================
    # 创建环境实例
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # ==================== 机器人外观设置 ====================
    # 设置所有刚体为灰色，便于观察
    for i in range(env.num_bodies):
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
    
    # ==================== 策略加载 ====================
    # 加载训练好的策略
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # ==================== 相机控制配置 ====================
    cfg_eval = {
        "timesteps": (env_cfg.env.episode_length_s) * 500 + 1,  # 总时间步数
        'cameraTrack': True,           # 启用相机自动跟踪
        'trackIndex': 0,               # 跟踪第0个环境中的机器人
        "cameraInit": np.pi*8/10,      # 初始相机角度
        "cameraVel": 1*np.pi/10,       # 相机旋转速度
    }
    
    # 相机控制参数
    camera_rot = np.pi * 8 / 10                    # 初始相机旋转角度
    camera_rot_per_sec = 1 * np.pi / 10            # 相机每秒旋转角度
    camera_relative_position = np.array([1, 0, 0.8])  # 相机相对机器人位置
    track_index = 0                                # 跟踪的机器人索引

    # ==================== 相机初始设置 ====================
    # 获取机器人位置作为相机焦点
    look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
    # 设置相机位置和焦点
    env.set_camera(look_at + camera_relative_position, look_at, track_index)
    
    # ==================== 环境重置和初始化 ====================
    # 重置环境
    _, _ = env.reset()
    # 执行一步，获取初始观察
    obs, critic_obs, _, _, _ = env.step(torch.zeros(
            env.num_envs, env.num_actions, dtype=torch.float, device=env.device))

    # ==================== 主循环：策略执行和相机控制 ====================
    timesteps = env_cfg.env.episode_length_s * 500 + 1
    for timestep in tqdm.tqdm(range(timesteps)):
        # 使用推理模式，不计算梯度
        with torch.inference_mode():
            # 使用策略生成动作
            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)

            # 执行动作，获取新的观察
            obs, critic_obs, _, _, _ = env.step(actions)
            
            # ==================== 相机动态跟踪 ====================
            # 获取当前机器人位置作为相机焦点
            look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
            # 更新相机旋转角度
            camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
            
            # 相机运动参数
            h_scale = 1      # 水平缩放
            v_scale = 0.8    # 垂直缩放
            
            # 计算相机相对位置（围绕机器人旋转）
            camera_relative_position = 2 * \
                np.array([np.cos(camera_rot) * h_scale,
                         np.sin(camera_rot) * h_scale, 0.5 * v_scale])
            
            # 更新相机位置
            env.set_camera(look_at + camera_relative_position, look_at, track_index)

            # ==================== 运动命令设置 ====================
            # 设置机器人运动命令
            env.commands[:, 0] = 2.0   # 前进速度 (m/s)
            env.commands[:, 1] = 0     # 侧向速度 (m/s)
            env.commands[:, 2] = 0     # 角速度 (rad/s)
            env.commands[:, 3] = 2.0   # 目标高度 (m)
            env.commands[:, 4] = 0.5   # 其他参数
            env.commands[:, 5] = 0.5   # 其他参数
            env.commands[:, 6] = 0.2   # 其他参数
            env.commands[:, 7] = -0.0  # 其他参数
            env.commands[:, 8] = 0.0   # 其他参数
            env.commands[:, 9] = 0.0   # 其他参数
            
            # ==================== 干扰和中断设置 ====================
            # 启用干扰功能
            env.use_disturb = True
            # 设置干扰掩码（所有关节都启用干扰）
            env.disturb_masks[:] = True
            # 设置干扰类型为噪声
            env.disturb_isnoise[:]= True
            # 设置干扰半径课程学习值
            env.disturb_rad_curriculum[:] = 1.0
            # 设置中断掩码
            env.interrupt_mask[:] = env.disturb_masks[:]
            # 设置所有环境为站立模式
            env.standing_envs_mask[:] = True
            # 站立时停止运动（前3个命令设为0）
            env.commands[env.standing_envs_mask, :3] = 0

if __name__ == '__main__':
    args = get_args()
    play(args)