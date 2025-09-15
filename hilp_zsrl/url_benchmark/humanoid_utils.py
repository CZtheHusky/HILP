import os
import matplotlib
matplotlib.use("Agg", force=True)  # 双保险
import matplotlib.pyplot as plt
# NumPy compat shim for legacy Isaac Gym utils (np.float deprecation)
import numpy as _np
if not hasattr(_np, 'float'):
    _np.float = float  # type: ignore[attr-defined]
import numpy as np
from matplotlib.colors import Normalize

def _safe_set_viewer_cam(env, pos, look, track_index=0):
    env.set_camera(pos, look, track_index)

# calculate the cosine similarity of a trajectory
def get_cosine_sim(array: np.ndarray) -> np.ndarray:
    assert len(array.shape) == 2
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)  # 防止除零
    unit = array / norms
    return unit @ unit.T

def collect_nonzero_losses(M: np.ndarray, eps: float = 0.0):
    """
    从 φ-loss 矩阵中收集非零（或绝对值>eps）的有效项（忽略 NaN），返回数值列表。
    """
    mask = ~np.isnan(M)
    if eps > 0:
        mask &= (np.abs(M) > eps)
    else:
        mask &= (M != 0)
    return M[mask].tolist()

def collect_nonzero_losses_with_idx(M: np.ndarray, eps: float = 0.0):
    """
    收集 (i, g, value) 三元组：i 行、g 列、对应 loss 值。
    仅保留有效且非零（或绝对值>eps）的条目。
    """
    mask = ~np.isnan(M)
    if eps > 0:
        mask &= (np.abs(M) > eps)
    else:
        mask &= (M != 0)
    is_, gs = np.where(mask)
    return [(int(i), int(g), float(M[i, g])) for i, g in zip(is_, gs)]


def calc_phi_loss_upper(arr: np.ndarray, gamma: float = 0.99, fill_value: float = np.nan) -> np.ndarray:
    """
    计算 φ-loss：
        L(i,g) = -1 - gamma * ||arr[i+1] - arr[g]|| + ||arr[i] - arr[g]||, 仅对 g>i

    返回 T x T 的严格上三角矩阵 M：
        M[i, g] = L(i, g)  (仅 g>i 有值)
        其它位置填 fill_value（默认 NaN）
    """
    assert arr.ndim == 2, f"expect 2D, got {arr.shape}"
    T = arr.shape[0]
    if T < 2:
        return np.full((T, T), fill_value, dtype=np.float32)

    # 所有 pairwise L2 距离 dists[a, b] = ||arr[a] - arr[b]||
    diffs = arr[:, None, :] - arr[None, :, :]   # [T, T, D]
    dists = np.linalg.norm(diffs, axis=-1)      # [T, T]

    # 严格上三角索引：rows=i, cols=g, 且 g>i
    rows, cols = np.triu_indices(T, k=1)

    out = np.full((T, T), fill_value, dtype=dists.dtype)
    # 注意：rows 最大为 T-2，因此 rows+1 索引安全
    out[rows, cols] = -1.0 - gamma * dists[rows + 1, cols] + dists[rows, cols]
    return out


def calc_matrix(diffs, fill_value):
    arr_len = diffs.shape[0]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)    # (g, 1)

    # ---- 绝对距离 d(i,g) 及其差分矩阵 ----
    dists = norms.squeeze(-1)                               # (g,)
    valid_d = np.isfinite(dists)
    D = dists[:, None] - dists[None, :]                     # (g, g)
    vv_d = np.logical_and.outer(valid_d, valid_d)           # (g, g)
    D[~vv_d] = fill_value

    # 保存绝对距离向量（无效处设为 NaN，便于绘图）
    d_plot = dists.copy()
    d_plot[~valid_d] = np.nan

    # ---- 余弦相似度 ----
    valid = valid_d & (dists > 0)
    norms_safe = np.where(valid[:, None], norms, 1.0)
    Z = diffs / norms_safe
    Z[~valid] = 0.0
    G = Z @ Z.T

    M = np.full((arr_len, arr_len), fill_value, dtype=float)
    vv = np.logical_and.outer(valid, valid)
    np.fill_diagonal(vv, False)
    M[vv] = G[vv]
    return M, D, d_plot

def calc_z_vector_matrixes(arr: np.ndarray, fill_value: float = np.nan, init_chunk_size=10, goal_vector=None):
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]

    L = []           # 余弦相似度矩阵列表
    Distance = []    # 距离差矩阵列表
    AbsDists = []    # 每个 g 的绝对距离向量 d(i,g)

    g = init_chunk_size
    break_flag = False
    assert len(arr.shape) == 2, f"expect 2D, got {arr.shape}"
    if goal_vector is None:
        while g < n:
            diffs = arr[g] - arr[:g + 1]                                # (g, d)
            M, D, d_plot = calc_matrix(diffs, fill_value)
            Distance.append(D)
            AbsDists.append(d_plot)
            L.append(M)
            # 递增 g：×4，并在最后一次补到 n-1
            g = g * 4
            if break_flag:
                break
            if g > n - 1:
                g = n - 1
                break_flag = True
    else:
        diffs = goal_vector - arr
        M, D, d_plot = calc_matrix(diffs, fill_value)
        Distance.append(D)
        AbsDists.append(d_plot)
        L.append(M)
    return L, Distance, AbsDists



def compute_global_color_limits(mats, lower_q: float = 5, upper_q: float = 95):
    """
    参数:
        mats: 由多个 (T, T) φ-loss 矩阵构成的列表，矩阵中无效处为 NaN
        lower_q, upper_q: 用分位数裁剪极端值，稳健设定颜色范围
    返回:
        (vmin, vmax)
    """
    vals = []
    for M in mats:
        if M is None:
            continue
        v = M[~np.isnan(M)]
        if v.size:
            vals.append(v)
    if not vals:
        return None, None
    all_vals = np.concatenate(vals)
    vmin, vmax = np.percentile(all_vals, [lower_q, upper_q])
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmin == vmax:
        eps = 1e-6
        vmin -= eps
        vmax += eps
    return float(vmin), float(vmax)


def plot_matrix_heatmaps(
    mats,
    names,
    out_dir: str,
    vmin,
    vmax,
    max_T=None,
    cmap_name: str = "coolwarm",
    label: str = "φ-loss",
    title: str = "Phi loss (upper triangle)",
    xlabel: str = "j (heatmap)",
    ylabel: str = "i (heatmap)",
):
    """
    对每个矩阵生成一个图像文件：
      - 若没有对应的 line_series，则仅包含热力图；
      - 若有对应的 line_series，则在下方额外添加一个共享 x 轴的折线子图。
    """
    os.makedirs(out_dir, exist_ok=True)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#eeeeee")  # NaN 区域浅灰

    for idx, (M, name) in enumerate(zip(mats, names)):
        if M is None:
            continue

        M_plot = M[:max_T, :max_T] if max_T is not None else M
        g_i, g_j = M_plot.shape
        # 如果想象是方阵，通常 g_i == g_j；此处按列数对齐 x 轴
        n_cols = g_j

        masked = np.ma.masked_invalid(M_plot)

        fig, axs = plt.subplots(figsize=(6, 5), dpi=150)
        # ---- 热力图 ----
        # 通过 extent 确保像素中心落在整数 j 上，并让 y 自上而下
        im = axs.imshow(
            masked,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
            extent=(-0.5, n_cols - 0.5, g_i - 0.5, -0.5),  # x: -0.5..n-0.5, y: g-0.5..-0.5
            aspect="auto",
        )
        cb = fig.colorbar(im, ax=axs)
        cb.set_label(label)

        axs.set_title(f"{title} - {name}")
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)

        # x 轴范围与像素边界严格一致
        axs.set_xlim(-0.5, n_cols - 0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, name), bbox_inches="tight")
        plt.close(fig)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_tripanel_heatmaps_with_line(
    cos_mats,
    dist_mats,
    line_series,
    names,
    out_dir: str,
    max_T=None,
    cmap_cos: str = "coolwarm",
    cmap_dist: str = "viridis",
    cos_label: str = "goal_z_cos_sim",
    dist_label: str = "latent_space_distance",
    line_label: str = "d(i,g)",
    xlabel: str = "j",
    ylabel: str = "i",
    title_cos: str = "Goal z cosine similarity",
    title_dist: str = "Latent space distance",
):
    """
    3 行 × 1 列：每行一个正方形主轴。
    对第 1、2 行，用 inset_axes 在主轴右侧放“短色条”（不再占用独立网格列）。
    """
    os.makedirs(out_dir, exist_ok=True)

    cos_vmin, cos_vmax = compute_global_color_limits(cos_mats)
    dist_vmin, dist_vmax = compute_global_color_limits(dist_mats)

    num = max(len(cos_mats) if cos_mats is not None else 0,
              len(dist_mats) if dist_mats is not None else 0,
              len(line_series) if line_series is not None else 0)

    for idx in range(num):
        Mc = None if cos_mats is None or idx >= len(cos_mats) else cos_mats[idx]
        Md = None if dist_mats is None or idx >= len(dist_mats) else dist_mats[idx]
        dvec = None if line_series is None or idx >= len(line_series) else line_series[idx]
        if Mc is None and Md is None and dvec is None:
            continue

        n_cols_candidates = []
        if Mc is not None:
            Mc = Mc[:max_T, :max_T] if max_T is not None else Mc
            n_cols_candidates.append(Mc.shape[1])
        if Md is not None:
            Md = Md[:max_T, :max_T] if max_T is not None else Md
            n_cols_candidates.append(Md.shape[1])
        if dvec is not None:
            dvec = dvec[:max_T] if max_T is not None else dvec
            n_cols_candidates.append(len(dvec))
        if not n_cols_candidates:
            continue

        n_cols = min(n_cols_candidates)
        if Mc is not None: Mc = Mc[:n_cols, :n_cols]
        if Md is not None: Md = Md[:n_cols, :n_cols]
        if dvec is not None: dvec = dvec[:n_cols]
        if Mc is None: Mc = np.full((0, n_cols), np.nan)
        if Md is None: Md = np.full((0, n_cols), np.nan)
        if dvec is None: dvec = np.array([])

        Mc_masked = np.ma.masked_invalid(Mc.astype(float))
        Md_masked = np.ma.masked_invalid(Md.astype(float))
        gi_c = Mc.shape[0]
        gi_d = Md.shape[0]

        # ---- 单列 3 行；右侧不再预留色条列 ----
        side = 5.2  # 每个正方形轴的边长（可按需调）
        fig, axs = plt.subplots(
            3, 1, figsize=(side * 1.18, side * 3.1), dpi=150,  # 稍加一点右侧余量给嵌入色条
            sharex=True, constrained_layout=True
        )
        ax0, ax1, ax2 = axs

        # (0) Cosine heatmap
        im0 = ax0.imshow(
            Mc_masked, cmap=cmap_cos, vmin=cos_vmin, vmax=cos_vmax, interpolation="nearest",
            extent=(-0.5, n_cols - 0.5, gi_c - 0.5, -0.5)
        )
        ax0.set_title(title_cos)
        ax0.set_ylabel(ylabel)
        ax0.set_xlim(-0.5, n_cols - 0.5)
        ax0.set_box_aspect(1)  # 正方形

        # —— 在右侧嵌入“短色条”（高度 75%），不改变网格布局，不拉伸间距 ——
        cax0 = inset_axes(
            ax0, width="3.2%", height="100%",
            loc="center left", bbox_to_anchor=(1.02, 0, 1, 1),
            bbox_transform=ax0.transAxes, borderpad=0.0
        )
        cb0 = fig.colorbar(im0, cax=cax0)
        cb0.set_label(cos_label)
        cax0.yaxis.set_label_position("right")
        cax0.yaxis.tick_right()

        # (1) Distance heatmap
        im1 = ax1.imshow(
            Md_masked, cmap=cmap_dist, vmin=dist_vmin, vmax=dist_vmax, interpolation="nearest",
            extent=(-0.5, n_cols - 0.5, gi_d - 0.5, -0.5)
        )
        ax1.set_title(title_dist)
        ax1.set_ylabel(ylabel)
        ax1.set_xlim(-0.5, n_cols - 0.5)
        ax1.set_box_aspect(1)  # 正方形

        cax1 = inset_axes(
            ax1, width="3.2%", height="100%",
            loc="center left", bbox_to_anchor=(1.02, 0, 1, 1),
            bbox_transform=ax1.transAxes, borderpad=0.0
        )
        cb1 = fig.colorbar(im1, cax=cax1)
        cb1.set_label(dist_label)
        cax1.yaxis.set_label_position("right")
        cax1.yaxis.tick_right()

        # (2) line: d(i,g)（同样保持正方形绘图区）
        x = np.arange(len(dvec))
        if len(x):
            ax2.plot(x, dvec, marker="o", linewidth=1.5)
        ax2.set_ylabel(line_label)
        ax2.set_xlabel(xlabel)
        ax2.set_xlim(-0.5, n_cols - 0.5)
        if n_cols > 20:
            step = max(1, n_cols // 10)
            ax2.set_xticks(np.arange(0, n_cols, step))
        ax2.set_box_aspect(1)

        # 保存
        name = names[idx] if idx < len(names) else f"g_{idx}.png"
        base = os.path.basename(name)
        if "." not in base:
            name = name + ".png"
        fig.savefig(os.path.join(out_dir, name))
        plt.close(fig)

def plot_per_step_z(
    latent,
    eid,
    out_dir,
    command_name=None,
    cos_raw_diff=None,
    cos_raw=None,
    raw_diff_distance=None,
    raw_distance=None,
):
    os.makedirs(out_dir, exist_ok=True)

    # --- 计算相似度 & 可视化 ---
    cos_latent = get_cosine_sim(latent)
    latent_diff = np.diff(latent, axis=0)
    cos_latent_diff = get_cosine_sim(latent_diff)

    latent_diff_distance = np.linalg.norm(latent_diff, axis=-1)
    latent_distance      = np.linalg.norm(latent, axis=-1)

    common_hm = dict(
        origin="lower", aspect="auto", interpolation="nearest",
        cmap="coolwarm", norm=Normalize(vmin=-1, vmax=1)
    )

    # 更健壮：只要任意 raw 相关数据存在，就走 4x2 布局
    has_raw = any(x is not None for x in (cos_raw, cos_raw_diff, raw_distance, raw_diff_distance))

    if has_raw:
        fig, axs = plt.subplots(4, 2, figsize=(12, 16), dpi=300, constrained_layout=True)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=300, constrained_layout=True)

    # ---- latent heatmaps ----
    im0 = axs[0, 0].imshow(cos_latent, **common_hm)
    axs[0, 0].set_title('cosine similarity (latent)')
    axs[0, 0].set_xlabel('time step'); axs[0, 0].set_ylabel('time step')

    im1 = axs[0, 1].imshow(cos_latent_diff, **common_hm)
    axs[0, 1].set_title('cosine similarity of Δ latent')
    axs[0, 1].set_xlabel('time step'); axs[0, 1].set_ylabel('time step')

    # ---- latent lines (修正 ylabel 归属) ----
    axs[1, 0].plot(latent_distance)
    axs[1, 0].set_title('‖latent‖')
    axs[1, 0].set_xlabel('time step'); axs[1, 0].set_ylabel('distance')

    axs[1, 1].plot(latent_diff_distance)
    axs[1, 1].set_title('‖Δ latent‖ (per step)')
    axs[1, 1].set_xlabel('time step'); axs[1, 1].set_ylabel('distance')

    # ---- raw 部分（若提供）----
    heatmap_axes = [axs[0, 0], axs[0, 1]]  # 用于色条绑定的热图轴列表

    if has_raw:
        # heatmaps（若缺某个，就隐藏对应轴）
        if cos_raw is not None:
            im2 = axs[2, 0].imshow(cos_raw, **common_hm)
            axs[2, 0].set_title('cosine similarity (raw state)')
            axs[2, 0].set_xlabel('time step'); axs[2, 0].set_ylabel('time step')
            heatmap_axes.append(axs[2, 0])
        else:
            axs[2, 0].axis('off')

        if cos_raw_diff is not None:
            im3 = axs[2, 1].imshow(cos_raw_diff, **common_hm)
            axs[2, 1].set_title('cosine similarity of Δ raw state')
            axs[2, 1].set_xlabel('time step'); axs[2, 1].set_ylabel('time step')
            heatmap_axes.append(axs[2, 1])
        else:
            axs[2, 1].axis('off')

        # lines（若缺某个，就隐藏对应轴）
        if raw_distance is not None:
            axs[3, 0].plot(raw_distance)
            axs[3, 0].set_title('‖raw state‖')
            axs[3, 0].set_xlabel('time step'); axs[3, 0].set_ylabel('distance')
        else:
            axs[3, 0].axis('off')

        if raw_diff_distance is not None:
            axs[3, 1].plot(raw_diff_distance)
            axs[3, 1].set_title('‖Δ raw state‖ (per step)')
            axs[3, 1].set_xlabel('time step'); axs[3, 1].set_ylabel('distance')
        else:
            axs[3, 1].axis('off')

    # ---- colorbar：仅绑定到实际的热图轴，不包含折线轴 ----
    fig.colorbar(im0, ax=heatmap_axes, location='right', shrink=0.9, label='cosine similarity')

    fig.suptitle(f'episode {eid}')
    save_path = (
        f"{out_dir}/cos_sim_{command_name}_{eid}.png"
        if command_name is not None else
        f"{out_dir}/cos_sim_{eid}.png"
    )
    # 用 constrained_layout 就别再 bbox_inches='tight'，更稳
    fig.savefig(save_path)
    plt.close(fig)



