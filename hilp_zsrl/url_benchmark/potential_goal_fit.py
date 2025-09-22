# fit_goal_latent.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Literal, Dict


@dataclass
class FitReport:
    z_star: np.ndarray              # (d,) 最终拟合到的 latent goal（未标准化坐标系）
    train_mse: float
    val_mse: Optional[float]
    method: str
    optimizer: str
    num_iters: int
    restarts: int
    used_standardize: bool
    z_star_stdspace: Optional[np.ndarray]  # 若做了标准化，这是标准化空间下的z
    meta: Dict


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.ndim == 1:
        t = t[None, :]
    return t


def _split_train_val(N: int, val_ratio: float, rng: np.random.RandomState):
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(round(N * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def _standardize_features(x: torch.Tensor, xp: torch.Tensor, idx: np.ndarray):
    """只用训练集统计量做标准化；返回标准化后的x/xp以及均值方差"""
    mu = x[idx].mean(dim=0, keepdim=True)
    std = x[idx].std(dim=0, keepdim=True).clamp_min(1e-12)
    x_s = (x - mu) / std
    xp_s = (xp - mu) / std
    return x_s, xp_s, mu.squeeze(0), std.squeeze(0)


def _pred_diff_norm(z: torch.Tensor, x: torch.Tensor, xp: torch.Tensor, eps: float):
    # 预测 \|z-x\| - \|z-x'\|
    d1 = torch.sqrt(((z - x) ** 2).sum(dim=1) + eps)
    d2 = torch.sqrt(((z - xp) ** 2).sum(dim=1) + eps)
    return d1 - d2


def _loss(z, x, xp, r, lam, eps, idx):
    pred = _pred_diff_norm(z, x[idx], xp[idx], eps)
    mse = torch.mean((r[idx] - pred) ** 2)
    reg = lam * torch.sum(z ** 2)
    return mse + reg, mse


def _lstsq_init(x: torch.Tensor, xp: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    平方距离变体闭式解初始化：
    y = r - (||x||^2 - ||xp||^2)  ≈  2(xp - x)^T z
    => z = lstsq( 2(xp-x), y )
    """
    X = 2.0 * (xp - x)                       # (N, d)
    y = r - (x.pow(2).sum(1) - xp.pow(2).sum(1))  # (N,)
    # torch.linalg.lstsq 在 PyTorch>=1.9 可用
    sol = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze(1)  # (d,)
    return sol


def fit_goal_latent(
    x: np.ndarray,
    x_next: np.ndarray,
    r: np.ndarray,
    *,
    lam: float = 1e-3,
    eps: float = 1e-6,
    standardize: bool = True,
    val_ratio: float = 0.05,
    optimizer: Literal["lbfgs", "adam"] = "lbfgs",
    lbfgs_lr: float = 1.0,
    lbfgs_max_iter: int = 200,
    adam_lr: float = 1e-2,
    adam_epochs: int = 30,
    batch_size: int = 8192,
    restarts: int = 4,
    use_lstsq_init: bool = True,
    seed: int = 0,
    device: Optional[str] = None,
) -> FitReport:
    """
    拟合 latent goal z* 使得  r ≈ ||z-phi(s)|| - ||z-phi(s')||  的 MSE 最小。

    参数
    ----
    x, x_next : (N, d)  latent features (phi(s), phi(s'))
    r         : (N,)    target rewards
    optimizer : "lbfgs" 适合中小N；"adam" 支持大N（mini-batch）
    restarts  : 多重重启，取验证集最优
    """
    assert x.shape == x_next.shape
    N, d = x.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.RandomState(seed)
    train_idx, val_idx = _split_train_val(N, val_ratio, rng)
    has_val = len(val_idx) > 0

    x_t = _to_tensor(x).to(device)
    xp_t = _to_tensor(x_next).to(device)
    r_t = _to_tensor(r).to(device).squeeze(0)

    # 标准化（可选）
    if standardize:
        x_s, xp_s, mu, std = _standardize_features(x_t, xp_t, train_idx)
        x_use, xp_use = x_s, xp_s
        mu_np, std_np = mu.detach().cpu().numpy(), std.detach().cpu().numpy()
    else:
        x_use, xp_use = x_t, xp_t
        mu_np, std_np = None, None

    best = {
        "val_mse": float("inf"),
        "train_mse": float("inf"),
        "z_std": None,
        "iters": 0,
        "method": "direct_norm",
        "optimizer": optimizer,
        "restart": -1,
    }

    # 生成初始化
    z0_list = []
    if use_lstsq_init:
        with torch.no_grad():
            z0 = _lstsq_init(x_use, xp_use, r_t).detach()
        z0_list.append(z0)
    # 额外随机初始化
    for _ in range(max(0, restarts - len(z0_list))):
        z0_list.append(torch.randn(d, device=device) * 0.1)

    for k, z0 in enumerate(z0_list):
        z = torch.nn.Parameter(z0.clone())
        if optimizer == "lbfgs":
            opt = torch.optim.LBFGS([z], lr=lbfgs_lr, max_iter=lbfgs_max_iter, line_search_fn="strong_wolfe")

            itercount = {"n": 0}

            def closure():
                opt.zero_grad(set_to_none=True)
                loss, _ = _loss(z, x_use, xp_use, r_t, lam, eps, train_idx)
                loss.backward()
                itercount["n"] += 1
                return loss

            opt.step(closure)

            with torch.no_grad():
                _, train_mse = _loss(z, x_use, xp_use, r_t, lam=0.0, eps=eps, idx=train_idx)
                if has_val:
                    pred_val = _pred_diff_norm(z, x_use[val_idx], xp_use[val_idx], eps)
                    val_mse = torch.mean((r_t[val_idx] - pred_val) ** 2)
                else:
                    val_mse = train_mse

            train_mse_f = float(train_mse.detach().cpu())
            val_mse_f = float(val_mse.detach().cpu())

            if val_mse_f < best["val_mse"]:
                best.update({"val_mse": val_mse_f, "train_mse": train_mse_f, "z_std": z.detach().clone(),
                             "iters": itercount["n"], "restart": k})

        elif optimizer == "adam":
            opt = torch.optim.Adam([z], lr=adam_lr)
            iters = 0
            for epoch in range(adam_epochs):
                # mini-batch over train_idx
                perm = train_idx.copy()
                rng.shuffle(perm)
                for bs in range(0, len(perm), batch_size):
                    idx = perm[bs:bs + batch_size]
                    opt.zero_grad(set_to_none=True)
                    loss, _ = _loss(z, x_use, xp_use, r_t, lam, eps, idx)
                    loss.backward()
                    opt.step()
                    iters += 1

            with torch.no_grad():
                _, train_mse = _loss(z, x_use, xp_use, r_t, lam=0.0, eps=eps, idx=train_idx)
                if has_val:
                    pred_val = _pred_diff_norm(z, x_use[val_idx], xp_use[val_idx], eps)
                    val_mse = torch.mean((r_t[val_idx] - pred_val) ** 2)
                else:
                    val_mse = train_mse

            train_mse_f = float(train_mse.detach().cpu())
            val_mse_f = float(val_mse.detach().cpu())
            if val_mse_f < best["val_mse"]:
                best.update({"val_mse": val_mse_f, "train_mse": train_mse_f, "z_std": z.detach().clone(),
                             "iters": iters, "restart": k})
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # 反标准化回原空间
    z_std = best["z_std"]  # (d,)
    if z_std is None:
        raise RuntimeError("Optimization failed to produce a solution.")

    if standardize:
        # 原空间： z_real = z_std * std + mu
        z_real = (z_std * torch.as_tensor(std_np, device=z_std.device)) + torch.as_tensor(mu_np, device=z_std.device)
        z_real = z_real
        z_std = z_std
    else:
        z_real = z_std
        z_std = None

    return z_real, {"train_mse": best["train_mse"], "val_mse": best["val_mse"] if has_val else None, "iters": best["iters"], "restart": best["restart"]}
    # FitReport(
    #     z_star=z_real_np,
    #     train_mse=best["train_mse"],
    #     val_mse=best["val_mse"] if has_val else None,
    #     method="||z-x|| - ||z-x'|| (least squares)",
    #     optimizer=optimizer,
    #     num_iters=best["iters"],
    #     restarts=len(z0_list),
    #     used_standardize=standardize,
    #     z_star_stdspace=z_std_np,
    #     meta=dict(
    #         lam=lam, eps=eps, seed=seed, lbfgs_max_iter=lbfgs_max_iter, adam_epochs=adam_epochs,
    #         batch_size=batch_size, restart_chosen=best["restart"], use_lstsq_init=use_lstsq_init
    #     )
    # )


# -------------------------- Demo --------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    N, d = 20000, 16
    # 构造一个“真”目标 z_true，并生成 (x, x_next, r)
    z_true = rng.randn(d)

    x = rng.randn(N, d)
    xp = x + 0.1 * rng.randn(N, d)  # 让 s' 靠近/远离一些
    # 真正的目标：r = ||z-x|| - ||z-x'||
    r = np.linalg.norm(z_true - x, axis=1) - np.linalg.norm(z_true - xp, axis=1)
    # 加噪声
    r += 0.05 * rng.randn(N)

    rep = fit_goal_latent(
        x, xp, r,
        standardize=True,
        optimizer="adam",        # 大数据建议adam；小数据可换"lbfgs"
        adam_lr=5e-3,
        adam_epochs=15,
        batch_size=4096,
        restarts=4,
        use_lstsq_init=True,
        lam=1e-3,
        eps=1e-6,
        seed=0,
    )

    print("=== Fit Report ===")
    print(f"train_mse: {rep.train_mse:.6f}")
    print(f"val_mse  : {rep.val_mse}")
    print(f"iters    : {rep.num_iters}, optimizer: {rep.optimizer}, restarts: {rep.restarts}")
    print(f"z* (first 5 dims): {rep.z_star[:5]}")
    print(f"|z* - z_true|_2  : {np.linalg.norm(rep.z_star - z_true):.4f}")
