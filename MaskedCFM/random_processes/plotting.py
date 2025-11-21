import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


__all__ = ["plot_time_series", "plot_state_space", "plot_block_weight_heatmaps"]

##########################
# Plotting functions
##########################
def plot_time_series(
    time_data_pairs,
    labels=None,
    dims=None,
    plot_mean=False,
    plot_median=False,
    plot_ci=False,
    ci_level=2.0,
    true_mean=None,
    true_std=None,
    max_samples=None,
    separate_dims=False,
    figsize=(10, 4),
):
    """
    Flexible time-series plotter with optional per-dimension subplots.

    Parameters
    ----------
    time_data_pairs : list of tuples [(times_i, data_i), ...]
        times_i : array-like of shape (T_i,)
        data_i  : array-like of shape (N_i, T_i, d)
    labels : list of str, optional
    dims : list/tuple of ints, optional
    true_mean/true_std : array or list aligned with time_data_pairs (optional)
    """
    if labels is None:
        labels = [f"series_{i}" for i in range(len(time_data_pairs))]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    processed = []
    for times_i, data_i in time_data_pairs:
        t_arr = np.asarray(times_i)
        data_arr = np.asarray(data_i)
        processed.append((t_arr, data_arr))

    d = processed[0][1].shape[-1]
    if dims is None:
        dims = list(range(min(d, 3)))
    dims = np.atleast_1d(dims)

    def get_true_stat(stat, idx):
        if stat is None:
            return None
        if isinstance(stat, (list, tuple)):
            return np.asarray(stat[idx])
        return np.asarray(stat)

    def plot_dim(ax, dim_idx):
        for idx, ((times_i, data_i), label) in enumerate(zip(processed, labels)):
            color_i = colors[idx % len(colors)]
            N = data_i.shape[0]
            count = min(N, max_samples) if max_samples is not None else N
            for j in range(count):
                ax.plot(times_i, data_i[j, :, dim_idx], alpha=0.3, lw=0.7)

            if plot_mean:
                mean_i = data_i.mean(axis=0)
                ax.plot(times_i, mean_i[:, dim_idx], color=color_i, lw=2, label=f"{label}: sample mean")
                if plot_ci:
                    std_i = data_i.std(axis=0, ddof=1)
                    ax.fill_between(
                        times_i,
                        mean_i[:, dim_idx] - ci_level * std_i[:, dim_idx],
                        mean_i[:, dim_idx] + ci_level * std_i[:, dim_idx],
                        color=color_i,
                        alpha=0.2,
                        label=f"{label}: sample CI",
                    )

            if plot_median:
                median_i = np.median(data_i, axis=0)
                ax.plot(times_i, median_i[:, dim_idx], lw=2, color=color_i, linestyle="--", label=f"{label}: sample median")

            tm_i = get_true_stat(true_mean, idx)
            ts_i = get_true_stat(true_std, idx)

            if tm_i is not None:
                ax.plot(times_i, tm_i[:, dim_idx], color="tab:orange", lw=2, label=f"{label}: true mean")
                if ts_i is not None:
                    ax.fill_between(
                        times_i,
                        tm_i[:, dim_idx] - ci_level * ts_i[:, dim_idx],
                        tm_i[:, dim_idx] + ci_level * ts_i[:, dim_idx],
                        color="tab:orange",
                        alpha=0.15,
                        label=f"{label}: true ± CI",
                    )

        ax.set_ylabel(f"x_{dim_idx + 1}(t)")
        ax.grid(True, alpha=0.3)

    if separate_dims:
        fig, axes = plt.subplots(len(dims), 1, sharex=True,
                                 figsize=(figsize[0], figsize[1] * len(dims)))
        axes = np.atleast_1d(axes)
        for ax, dim_idx in zip(axes, dims):
            plot_dim(ax, dim_idx)
        axes[-1].set_xlabel("time")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        plot_dim(ax, dims[0])
        ax.set_xlabel("time")

    fig.suptitle("Time-series trajectories")
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.axes[0].legend(loc="best")
    plt.tight_layout()
    return fig


def plot_state_space(
    series_list,
    proj_dims=(0, 1),
    labels=None,
    max_samples=100,
    plot_mean=False,
    plot_median=False,
    plot_ci=False,
    ci_level=2.0,
    true_mean=None,
    true_cov=None,
    figsize=(6, 6),
):
    """
    State-space (2D/3D) plot for trajectories.

    Parameters
    ----------
    series_list : list of arrays (N, T, d)
    proj_dims : tuple
        Indices of coordinates to project onto (len 2 or 3).
    labels : list of str
        Legend labels.
    max_samples : int
        Number of sample trajectories per dataset to plot.
    plot_mean/median : bool
        Draw sample mean/median trajectory.
    plot_ci : bool
        Draw ellipsoidal confidence region around mean if true_cov provided.
    true_mean, true_cov : arrays
        For reference (optional). true_cov used only if plot_ci and len(proj_dims)==2.
    """
    #dims = np.atleast_1d(proj_dims)
    dims = [int(d) for d in np.atleast_1d(proj_dims)]

    if len(dims) not in (2, 3):
        raise ValueError("proj_dims must be pair or triple of indices.")

    if labels is None:
        labels = [f"series_{i}" for i in range(len(series_list))]

    fig = plt.figure(figsize=figsize)
    if len(dims) == 2:
        ax = fig.add_subplot(111)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # lazily import
        ax = fig.add_subplot(111, projection="3d")

    for data, label in zip(series_list, labels):
        data = np.asarray(data)
        N, T, d = data.shape
        count = min(N, max_samples) if max_samples is not None else N
        for i in range(count):
            if len(dims) == 2:
                ax.plot(data[i, :, dims[0]], data[i, :, dims[1]], alpha=0.4, label=label if i == 0 else "")
            else:  # 3D
                ax.plot(data[i, :, dims[0]], data[i, :, dims[1]], data[i, :, dims[2]], alpha=0.3)

        if plot_mean:
            mean = data.mean(axis=0)[:, dims]
            if len(dims) == 2:
                ax.plot(mean[:, 0], mean[:, 1], color="tab:orange", lw=2, label=f"{label} mean")
            else:
                ax.plot(mean[:, 0], mean[:, 1], mean[:, 2], color="tab:orange", lw=2)

        if plot_median:
            med = np.median(data, axis=0)[:, dims]
            if len(dims) == 2:
                ax.plot(med[:, 0], med[:, 1], color="tab:green", lw=2, label=f"{label} median")

    if true_mean is not None:
        tm = np.asarray(true_mean)[:, dims]
        if len(dims) == 2:
            ax.plot(tm[:, 0], tm[:, 1], color="tab:orange", lw=2, label="true mean")
        else:
            ax.plot(tm[:, 0], tm[:, 1], tm[:, 2], color="tab:orange", lw=2)

    ax.set_xlabel(f"x_{dims[0]+1}")
    ax.set_ylabel(f"x_{dims[1]+1}")

    if len(dims) == 3:
        ax.set_zlabel(f"x_{dims[2]+1}")
    ax.set_title("State-space trajectories")
    ax.grid(True, alpha=0.3)
    if len(dims) == 2:
        ax.legend(loc="best")
    plt.tight_layout()
    return fig


def plot_block_weight_heatmaps(
    model,
    mode="block",
    include_shared=False,
    cmap="magma",
    annotate=False,
    norm="fro",
):
    """
    Visualize masked weights as heatmaps.

    Parameters
    ----------
    model : MaskedBlockMLP
    mode : {"block", "full"}
        "block" shows T×T block magnitudes. "full" ignores block structure and plots
        each weight entry. Masked connections appear in black.
    include_shared : bool
        Include shared blocks when computing block magnitudes.
    """
    layers = getattr(model, "layers", None)
    layouts = getattr(model, "_block_layouts", None)
    if layers is None or layouts is None:
        raise TypeError("Model must expose .layers and _block_layouts (MaskedBlockMLP).")

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4), squeeze=False)

    if mode == "block":
        block_vals = []
        block_masks = []
        for layer, layout in zip(layers, layouts):
            in_blocks = list(layout["in_blocks"])
            if include_shared and layout["shared_block"]:
                in_blocks += list(layout["shared_block"])
            out_blocks = list(layout["out_blocks"])
            weight = layer.weight.detach()
            mask = layer.mask.detach()
            block_arr = torch.zeros(len(out_blocks), len(in_blocks))
            block_mask = torch.zeros(len(out_blocks), len(in_blocks))
            row_start = 0
            for r, out_dim in enumerate(out_blocks):
                col_start = 0
                for c, in_dim in enumerate(in_blocks):
                    w_block = weight[row_start:row_start + out_dim, col_start:col_start + in_dim]
                    m_block = mask[row_start:row_start + out_dim, col_start:col_start + in_dim]
                    if norm == "fro":
                        val = torch.linalg.norm(w_block, ord="fro")
                    elif norm == "absmax":
                        val = w_block.abs().max()
                    elif norm == "l1":
                        val = w_block.abs().sum()
                    else:
                        raise ValueError(f"Unsupported norm '{norm}'.")
                    block_arr[r, c] = val
                    block_mask[r, c] = float(m_block.any())
                    col_start += in_dim
                row_start += out_dim
            block_vals.append(block_arr.cpu().numpy())
            block_masks.append(block_mask.cpu().numpy())
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="black")
        for idx, (ax, arr, mask_arr) in enumerate(zip(axes[0], block_vals, block_masks)):
            vmax = arr.max() if arr.size else 1.0
            display = arr.copy()
            display[mask_arr == 0] = np.nan
            im = ax.imshow(display, cmap=cmap_obj, vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_title(f"Layer {idx + 1}")
            ax.set_xlabel("Input blocks")
            ax.set_ylabel("Output blocks")
            if annotate:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    elif mode == "full":
        processed = []
        for layer in layers:
            weight = layer.weight.detach().cpu().numpy()
            mask = layer.mask.detach().cpu().numpy()
            arr = np.abs(weight)
            processed.append((arr, mask))
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="black")
        for idx, (ax, (arr, mask)) in enumerate(zip(axes[0], processed)):
            vmax = arr.max() if arr.size else 1.0
            display = arr.copy()
            display[mask == 0] = np.nan
            im = ax.imshow(display, cmap=cmap_obj, vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_title(f"Layer {idx + 1}")
            ax.set_xlabel("Input neurons")
            ax.set_ylabel("Output neurons")
    else:
        raise ValueError("mode must be 'block' or 'full'")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    cbar.set_label("Magnitude")
    fig.suptitle("Masked weight heatmaps")
    fig.tight_layout()
    return fig




# def plot_gp_stats(times, true_mean, cov, samples, num_std=2):
#     """
#     Plot true vs. empirical mean ± num_std * std for each dimension over time.

#     times       : tensor/array shape (T,)
#     true_mean   : tensor/array shape (T, d)
#     cov         : full covariance matrix shape (T*d, T*d)
#     samples     : tensor/array shape (N, T, d)
#     num_std     : confidence band width (default ±2σ)
#     """
#     times = np.asarray(times)
#     true_mean = np.asarray(true_mean)
#     samples = np.asarray(samples)
#     N, T, d = samples.shape
#     cov_blocks = np.asarray(cov).reshape(T, d, T, d)

#     sample_mean = samples.mean(axis=0)
#     sample_std = samples.std(axis=0, ddof=1)
#     true_std = np.stack([np.sqrt(np.diag(cov_blocks[t, :, t, :])) for t in range(T)], axis=0)

#     fig, axes = plt.subplots(d, 1, figsize=(8, 3 * d), sharex=True)
#     axes = np.atleast_1d(axes)

#     for j in range(d):
#         ax = axes[j]
#         ax.plot(times, true_mean[:, j], color="tab:orange", lw=2, label="True mean")
#         ax.fill_between(times,
#                         true_mean[:, j] - num_std * true_std[:, j],
#                         true_mean[:, j] + num_std * true_std[:, j],
#                         color="tab:orange", alpha=0.2, label=f"True ± {num_std}σ" if j == 0 else None)

#         ax.plot(times, sample_mean[:, j], color="tab:orange", lw=2, label="Sample mean")
#         ax.fill_between(times,
#                         sample_mean[:, j] - num_std * sample_std[:, j],
#                         sample_mean[:, j] + num_std * sample_std[:, j],
#                         color="tab:orange", alpha=0.2, label=f"Sample ± {num_std}σ" if j == 0 else None)

#         ax.set_ylabel(f"x_{j+1}(t)")
#         ax.grid(True, alpha=0.3)
#         if j == 0:
#             ax.legend(loc="upper right")
#     axes[-1].set_xlabel("time")
#     plt.tight_layout()
#     plt.show()


# def plot_gp_trajectories(samples, dims=(0, 1), max_traj=100, alpha=0.4):
#     """
#     Plot sample paths in ℝᵈ projected onto the specified coordinate pair.

#     samples : tensor/array shape (N, T, d)
#     dims    : tuple of two coordinate indices to plot (default first two dims)
#     max_traj: maximum number of trajectories to show
#     alpha   : line transparency
#     """
#     samples = np.asarray(samples)
#     N, T, d = samples.shape
#     if len(dims) != 2:
#         raise ValueError("dims must be a pair of coordinate indices.")
#     if max(dims) >= d:
#         raise ValueError(f"dims {dims} exceed sample dimension {d}.")

#     fig, ax = plt.subplots(figsize=(6, 6))
#     n_show = min(max_traj, N)
#     for i in range(n_show):
#         ax.plot(samples[i, :, dims[0]], samples[i, :, dims[1]], alpha=alpha)
#     ax.set_xlabel(f"x_{dims[0]+1}")
#     ax.set_ylabel(f"x_{dims[1]+1}")
#     ax.set_title(f"{n_show} sample trajectories in ℝ^{d} (projected)")
#     ax.grid(True, alpha=0.3)
#     plt.show()
