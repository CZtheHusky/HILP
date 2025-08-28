import os
import sys
sys.path.append(os.getcwd())

import pickle

# ==================== 测试模式检测 ====================
# 早期读取测试模式，避免导入重型依赖
# 通过环境变量 ROLLOUT_TEST_MODE 控制是否启用测试模式
_TEST_MODE = os.environ.get("ROLLOUT_TEST_MODE", "0") == "1"

if not _TEST_MODE:
    # 非测试模式：导入完整的依赖
    from legged_gym.envs import *
    from legged_gym.utils import get_args, task_registry
    # 尝试延迟导入 Isaac Gym，这样测试模式可以在没有 Isaac Gym 的情况下运行
    try:
        import isaacgym  # type: ignore
        _HAS_ISAAC = True
    except Exception:
        _HAS_ISAAC = False


def _add_custom_args(args):
    """
    添加rollout特有的自定义参数
    
    legged_gym.utils.get_args() 已经解析了基础参数。
    我们通过环境变量或默认值来扩展rollout特有的参数。
    如果用户想要覆盖，可以在运行前设置环境变量。
    对于显式CLI，建议使用以下环境变量：
    ROLLOUT_TIMESTEPS, ROLLOUT_NUM_ENVS, ROLLOUT_OUTPUT, ROLLOUT_CKPT
    """
    # 设置rollout时间步数（默认5000步）
    args.rollout_timesteps = int(os.environ.get("ROLLOUT_TIMESTEPS", "5000"))
    # 设置rollout环境数量（默认1个环境）
    args.rollout_num_envs = int(os.environ.get("ROLLOUT_NUM_ENVS", "1"))
    # 设置输出文件路径（默认rollout_trajectory.npz）
    args.rollout_output = os.environ.get("ROLLOUT_OUTPUT", "rollout_trajectory.npz")
    # 设置检查点路径（可选）
    args.rollout_ckpt = os.environ.get("ROLLOUT_CKPT")
    # 设置是否启用测试模式
    args.rollout_test_mode = os.environ.get("ROLLOUT_TEST_MODE", "0") == "1"
    # 设置是否使用预设命令（默认启用）
    args.rollout_use_presets = os.environ.get("ROLLOUT_USE_PRESETS", "1") == "1"
    return args


def rollout(args):
    """
    执行rollout（轨迹回放）的主要函数
    
    Args:
        args: 包含所有rollout参数的参数对象
    """
    # ==================== 测试模式处理 ====================
    # 如果启用测试模式，运行最小化的干运行来验证保存管道，无需重型依赖
    if args.rollout_test_mode:
        # 测试模式：创建虚拟数据
        T = min(64, args.rollout_timesteps)  # 限制测试时间步数
        N = max(1, int(os.environ.get("ROLLOUT_TEST_NUM_ENVS", "1")))  # 测试环境数量
        
        # 创建虚拟观察数据
        obs = [[[0.0] * 16 for _ in range(N)] for _ in range(T)]
        critic_obs = [[[0.0] * 32 for _ in range(N)] for _ in range(T)]
        actions = [[[0.0] * 8 for _ in range(N)] for _ in range(T)]
        rewards = [[0.0 for _ in range(N)] for _ in range(T)]
        dones = [[False for _ in range(N)] for _ in range(T)]
        root_states = [[[0.0] * 13 for _ in range(N)] for _ in range(T)]
        
        # 组织输出数据
        out = {
            "obs": obs, 
            "critic_obs": critic_obs, 
            "actions": actions, 
            "rewards": rewards, 
            "dones": dones, 
            "root_states": root_states, 
            "dt": 0.01
        }
        
        # 保存为pickle格式
        out_path = args.rollout_output if args.rollout_output.endswith('.pkl') else args.rollout_output + '.pkl'
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(out, f)
        print(f"[TEST MODE] Saved dummy rollout (pickle) to {out_path}")
        return

    # ==================== 依赖检查 ====================
    # 检查Isaac Gym是否可用
    if not _HAS_ISAAC:
        raise ImportError("Isaac Gym is required for real rollout. Set ROLLOUT_TEST_MODE=1 for a dummy test, or install/activate Isaac Gym.")

    # 延迟导入numpy/torch，只在真正执行rollout时导入
    import numpy as np
    import torch

    # ==================== 环境配置获取 ====================
    # 从任务注册表获取环境配置和训练配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # ==================== 评估环境配置 ====================
    # 配置评估环境参数
    env_cfg.env.num_envs = args.rollout_num_envs  # 设置环境数量
    env_cfg.terrain.curriculum = False            # 关闭地形课程学习
    env_cfg.rewards.penalize_curriculum = False   # 关闭奖励课程学习

    # ==================== 环境构建 ====================
    # 构建环境实例
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ==================== 策略加载 ====================
    # 准备运行器并加载检查点
    train_cfg.runner.resume = True
    if args.rollout_ckpt is not None:
        train_cfg.runner.resume_path = args.rollout_ckpt  # 设置检查点路径
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)  # 获取推理策略

    # ==================== 轨迹数据缓冲区初始化 ====================
    # 为轨迹数据分配缓冲区
    T = args.rollout_timesteps  # 时间步数
    N = env.num_envs            # 环境数量
    
    # 创建各种数据缓冲区
    obs_buf = torch.zeros((T, N, env.num_partial_obs), dtype=torch.float32, device=env.device)           # 观察缓冲区
    critic_obs_buf = torch.zeros((T, N, env.num_obs), dtype=torch.float32, device=env.device)           # 评论者观察缓冲区
    act_buf = torch.zeros((T, N, env.num_actions), dtype=torch.float32, device=env.device)              # 动作缓冲区
    rew_buf = torch.zeros((T, N), dtype=torch.float32, device=env.device)                              # 奖励缓冲区
    done_buf = torch.zeros((T, N), dtype=torch.bool, device=env.device)                                # 完成状态缓冲区
    root_states_buf = torch.zeros((T, N, env.root_states.shape[-1]), dtype=torch.float32, device=env.device)  # 根状态缓冲区

    # ==================== 环境初始化和第一步 ====================
    # 重置环境
    _, _ = env.reset()
    # 执行一步，获取初始观察
    obs, critic_obs, rews, dones, infos = env.step(torch.zeros(N, env.num_actions, dtype=torch.float32, device=env.device))

    # ==================== 预设命令调度函数 ====================
    # 预设命令调度，覆盖不同的指令模式（针对h1int任务）
    def apply_command_presets(step_index: int):
        """
        应用预设命令调度
        
        Args:
            step_index: 当前时间步索引
        """
        if not args.rollout_use_presets:
            return
        if not hasattr(env, "commands"):
            return
            
        # 每6步循环一次预设命令
        preset_id = step_index % 6
        
        # h1int命令布局：[vx, vy, yaw, gait_freq, phase1, phase2/flag, swing_h, body_h, body_pitch, waist_roll, interrupt_flag?]
        
        # 预设0：前进行走
        if preset_id == 0:
            env.commands[:, 0] = 0.6; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.0      # 前进速度0.6m/s
            env.commands[:, 3] = 2.0; env.commands[:, 4] = 0.0; env.commands[:, 5] = 0.5; env.commands[:, 6] = 0.2  # 步态参数
            if env.commands.shape[1] > 7:
                env.commands[:, 7] = 0.0; env.commands[:, 8] = 0.0; env.commands[:, 9] = 0.0  # 身体姿态参数
                
        # 预设1：原地旋转
        elif preset_id == 1:
            env.commands[:, 0] = 0.0; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.6      # 角速度0.6rad/s
            env.commands[:, 3] = 2.5; env.commands[:, 4] = 0.5; env.commands[:, 5] = 0.5; env.commands[:, 6] = 0.25  # 步态参数
            if env.commands.shape[1] > 7:
                env.commands[:, 7] = -0.1; env.commands[:, 8] = 0.1; env.commands[:, 9] = 0.0  # 身体姿态参数
                
        # 预设2：前进+左转
        elif preset_id == 2:
            env.commands[:, 0] = 0.4; env.commands[:, 1] = 0.0; env.commands[:, 2] = -0.4     # 前进0.4m/s，左转0.4rad/s
            env.commands[:, 3] = 3.0; env.commands[:, 4] = 0.0; env.commands[:, 5] = 0.5; env.commands[:, 6] = 0.3  # 步态参数
            if env.commands.shape[1] > 7:
                env.commands[:, 7] = 0.05; env.commands[:, 8] = 0.0; env.commands[:, 9] = 0.2  # 身体姿态参数
                
        # 预设3：原地站立
        elif preset_id == 3:
            env.commands[:, 0] = 0.0; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.0      # 静止
            env.commands[:, 3] = 2.0; env.commands[:, 4] = 0.5; env.commands[:, 5] = 0.5; env.commands[:, 6] = 0.2  # 步态参数
            if env.commands.shape[1] > 7:
                env.commands[:, 7] = -0.2; env.commands[:, 8] = 0.3; env.commands[:, 9] = 0.0  # 身体姿态参数
                
        # 预设4：斜向行走
        elif preset_id == 4:
            env.commands[:, 0] = 0.2; env.commands[:, 1] = 0.2; env.commands[:, 2] = 0.0      # 斜向速度0.2m/s
            env.commands[:, 3] = 1.8; env.commands[:, 4] = 0.5; env.commands[:, 5] = 0.5; env.commands[:, 6] = 0.15  # 步态参数
            if env.commands.shape[1] > 7:
                env.commands[:, 7] = 0.0; env.commands[:, 8] = 0.0; env.commands[:, 9] = -0.2  # 身体姿态参数
                
        # 预设5：中断测试
        elif preset_id == 5:
            env.commands[:, 0] = 0.0; env.commands[:, 1] = 0.0; env.commands[:, 2] = 0.0      # 静止
            env.commands[:, 3] = 2.0; env.commands[:, 4] = 0.5; env.commands[:, 5] = 0.5; env.commands[:, 6] = 0.2  # 步态参数
            
            # 启用中断（如果可用）
            if hasattr(env, "interrupt_mask"):
                env.use_disturb = True
                env.disturb_masks[:] = True
                env.disturb_isnoise[:] = True
                env.interrupt_mask[:] = env.disturb_masks[:]
                
            # 设置中断标志
            if env.commands.shape[1] > 10:
                env.commands[:, 10] = 1.0

    # ==================== 主rollout循环 ====================
    # 执行主要的rollout循环
    for t in range(T):
        # 使用推理模式，不计算梯度
        with torch.inference_mode():
            # 使用策略生成动作
            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
            # 应用预设命令
            apply_command_presets(t)
            # 执行动作，获取新的观察
            obs, critic_obs, rews, dones, infos = env.step(actions)

        # ==================== 数据记录 ====================
        # 记录当前时间步的所有数据
        obs_buf[t] = obs                    # 观察数据
        critic_obs_buf[t] = critic_obs      # 评论者观察数据
        act_buf[t] = actions                # 动作数据
        rew_buf[t] = rews                   # 奖励数据
        done_buf[t] = dones                 # 完成状态数据
        root_states_buf[t] = env.root_states  # 根状态数据

    # ==================== 数据保存 ====================
    # 将数据移动到CPU并保存
    out = {
        "obs": obs_buf.cpu().numpy(),           # 观察数据
        "critic_obs": critic_obs_buf.cpu().numpy(),  # 评论者观察数据
        "actions": act_buf.cpu().numpy(),       # 动作数据
        "rewards": rew_buf.cpu().numpy(),       # 奖励数据
        "dones": done_buf.cpu().numpy(),        # 完成状态数据
        "root_states": root_states_buf.cpu().numpy(),  # 根状态数据
        "dt": float(env.dt),                   # 时间步长
    }

    # 创建输出目录并保存为压缩的numpy数组格式
    os.makedirs(os.path.dirname(args.rollout_output) or ".", exist_ok=True)
    np.savez_compressed(args.rollout_output, **out)
    print(f"Saved rollout to {args.rollout_output}")


if __name__ == "__main__":
    if _TEST_MODE:
        # ==================== 测试模式参数处理 ====================
        # 测试模式：创建最小化的参数对象，无需导入legged_gym
        class _Args:
            def __init__(self):
                self.task = os.environ.get("ROLLOUT_TASK", "h1int")                    # 任务名称
                self.rollout_timesteps = int(os.environ.get("ROLLOUT_TIMESTEPS", "64")) # 时间步数
                self.rollout_num_envs = int(os.environ.get("ROLLOUT_NUM_ENVS", "1"))    # 环境数量
                self.rollout_output = os.environ.get("ROLLOUT_OUTPUT", "rollout_test.pkl")  # 输出文件
                self.rollout_ckpt = None                                               # 检查点路径
                self.rollout_test_mode = True                                          # 测试模式标志
                self.rollout_use_presets = True                                        # 使用预设标志
        args = _Args()
        rollout(args)
    else:
        # ==================== 正常模式参数处理 ====================
        # 正常模式：获取完整参数并添加自定义参数
        args = get_args()
        args = _add_custom_args(args)
        rollout(args)


