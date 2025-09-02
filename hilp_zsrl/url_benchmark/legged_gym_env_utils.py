from isaacgym import gymutil
from legged_gym.envs import *  # registers tasks
from legged_gym.utils import task_registry
from argparse import Namespace
from isaacgym import gymapi
import numpy as np


def build_isaac_namespace(
    task_name: str,
    num_envs: int,
    headless: bool = False,
    compute_device_id: int = 0,
    graphics_device_id: int = 0,
    physics: str = "physx",     # or "flex"
    use_gpu: bool = True,
    use_gpu_pipeline: bool = True,
):
    # 可按你仓库的判断逻辑，既支持字符串也支持 gymapi 常量
    try:
        physics_engine = gymapi.SIM_PHYSX if physics.lower() == "physx" else gymapi.SIM_FLEX
    except Exception:
        physics_engine = physics.lower()  # 回退为字符串，很多 helper 也接受

    sim_device_type = "cuda" if use_gpu else "cpu"
    sim_device = f"{sim_device_type}:{compute_device_id}" if sim_device_type == "cuda" else "cpu"

    return Namespace(
        # 基本
        task=task_name,
        num_envs=int(num_envs),
        headless=bool(headless),

        # 物理/管线
        physics_engine=physics_engine,
        use_gpu=bool(use_gpu),
        use_gpu_pipeline=bool(use_gpu_pipeline),
        sim_device_type=sim_device_type,
        sim_device=sim_device,
        pipeline="gpu" if use_gpu_pipeline else "cpu",

        # 设备/线程
        compute_device_id=int(compute_device_id),
        graphics_device_id=int(graphics_device_id),
        num_threads=int(0),

        # 兼容位（部分 helper/registry 会探测）
        flex=False,
        physx=(physics.lower() == "physx"),
        slices=int(0),
        subscenes=int(0),
        seed=int(1),
        device=sim_device,                 # 某些仓库会读这个
        capture_video=False,
        force_render=not headless,         # 有些示例用它来强制创建图形上下文
    )


def _make_eval_env(device_id = 0):
    task_name = 'h1int'
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1000

    # prevent in-episode command resampling; we will control commands manually
    env_cfg.commands.resampling_time = 1000

    # ---- Build an argparse.Namespace for Isaac Gym / legged-gym helpers ----
    compute_device_id = device_id
    graphics_device_id = device_id
    headless = False

    args = build_isaac_namespace(task_name, env_cfg.env.num_envs, headless, compute_device_id, graphics_device_id)        
    # 地形和域随机化设置
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_load = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_link_props = False
    env_cfg.domain_rand.randomize_base_mass = False
    
    env_cfg.rewards.penalize_curriculum = False
    # 地形设置为平地
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.selected = True
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    env_cfg.terrain.terrain_kwargs = {
        "random_uniform": {
            "min_height": -0.00,
            "max_height": 0.00,
            "step": 0.005,
            "downsampled_scale": 0.2
        },
    }
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    return env


def _to_rgb_frame(img_any, H, W):
    # 统一拿到 numpy 数组
    if isinstance(img_any, np.ndarray):
        arr = img_any
    else:
        arr = np.frombuffer(img_any, np.uint8)

    # (H*W*4,) 一维 RGBA buffer
    if arr.ndim == 1 and arr.size == H * W * 4:
        rgba = arr.reshape(H, W, 4)
        return rgba[..., :3].copy()  # RGB

    # (H, W*4) 二维 RGBA 展平
    if arr.ndim == 2 and arr.shape[0] == H and arr.shape[1] == W * 4:
        rgba = arr.reshape(H, W, 4)
        return rgba[..., :3].copy()

    # (H, W, C) 理想三维
    if arr.ndim == 3 and arr.shape[0] == H and arr.shape[1] == W:
        C = arr.shape[2]
        if C >= 3:
            return arr[..., :3].copy()
        elif C == 1:
            ch = arr[..., 0]
            if ch.dtype != np.uint8:
                gmin, gmax = float(np.nanmin(ch)), float(np.nanmax(ch))
                if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax - gmin < 1e-12:
                    ch_u8 = np.zeros_like(ch, dtype=np.uint8)
                else:
                    ch_u8 = np.clip((ch - gmin) / (gmax - gmin) * 255.0, 0, 255).astype(np.uint8)
            else:
                ch_u8 = ch
            return np.stack([ch_u8, ch_u8, ch_u8], axis=-1)

    # (H, W) 灰度/深度
    if arr.ndim == 2 and arr.shape == (H, W):
        gray = arr
        if gray.dtype != np.uint8:
            gmin, gmax = float(np.nanmin(gray)), float(np.nanmax(gray))
            if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax - gmin < 1e-12:
                gray_u8 = np.zeros_like(gray, dtype=np.uint8)
            else:
                gray_u8 = np.clip((gray - gmin) / (gmax - gmin) * 255.0, 0, 255).astype(np.uint8)
        else:
            gray_u8 = gray
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

    # 兜底：尝试按 RGBA 解释
    if arr.size == H * W * 4:
        rgba = arr.reshape(H, W, 4)
        return rgba[..., :3].copy()

    raise RuntimeError(f"Unexpected camera image shape/dtype: shape={arr.shape}, dtype={arr.dtype}")
