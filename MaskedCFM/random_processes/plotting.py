import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Sequence, Dict
import torch
import torch.nn as nn

try:
    from ..models.masked_MLP import BlockLowerTriLinear  # when imported as package
except ImportError:
    try:
        from MaskedCFM.models.masked_MLP import BlockLowerTriLinear
    except ImportError:
        from models.masked_MLP import BlockLowerTriLinear

__all__ = [
    "plot_time_series",
    "plot_state_space",
    "plot_block_weight_heatmaps",
    "plot_recursive_predictions",
]


def _blue_red_positive_cmap():
    colors = [
        (0.0, "#0024FF"),
        (0.5, "#FFD600"),
        (1.0, "#FF0000"),
    ]
    return LinearSegmentedColormap.from_list("blue_red_positive", [c for _, c in colors])


def _is_block_lower_tri_layer(layer):
    return hasattr(layer, "assemble_weight") and hasattr(layer, "_block_meta")


def _get_layer_weight_tensor(layer):
    if _is_block_lower_tri_layer(layer):
        return layer.assemble_weight().detach()
    if hasattr(layer, "weight"):
        return layer.weight.detach()
    raise AttributeError(f"Layer {layer.__class__.__name__} does not expose a weight tensor.")


def _get_layer_bias_tensor(layer):
    bias = getattr(layer, "bias", None)
    if bias is None:
        return None
    return bias.detach()


def _connect_series(times, values, anchor_time, anchor_value, connect):
    times = np.asarray(times)
    values = np.asarray(values)
    if connect and anchor_time is not None and anchor_value is not None:
        times = np.concatenate([[anchor_time], times])
        values = np.concatenate([[anchor_value], values])
    return times, values


def _broadcast_param(value, length):
    if isinstance(value, (list, tuple)):
        if len(value) != length:
            raise ValueError(f"Expected parameter list of length {length}, got {len(value)}.")
        return list(value)
    if value is None:
        return [None] * length
    return [value] * length

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
    colors=None,
    line_style="-",
    marker_style=None,
    marker_every=None,
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
    plot_mean/median/ci : bool or list-like per series
    ci_level : float or list-like per series
    max_samples : int or list-like per series
    colors/line_style/marker_style/marker_every : single value or list per series
    true_mean/true_std : array or list aligned with time_data_pairs (optional)
    """
    if labels is None:
        labels = [f"series_{i}" for i in range(len(time_data_pairs))]
    processed = []
    for times_i, data_i in time_data_pairs:
        t_arr = np.asarray(times_i)
        data_arr = np.asarray(data_i)
        processed.append((t_arr, data_arr))

    n_series = len(processed)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_opts = _broadcast_param(colors, n_series)
    line_styles = _broadcast_param(line_style, n_series)
    marker_styles = _broadcast_param(marker_style, n_series)
    marker_every_opts = _broadcast_param(marker_every, n_series)
    plot_mean_opts = _broadcast_param(plot_mean, n_series)
    plot_median_opts = _broadcast_param(plot_median, n_series)
    plot_ci_opts = _broadcast_param(plot_ci, n_series)
    ci_levels = _broadcast_param(ci_level, n_series)
    max_samples_opts = _broadcast_param(max_samples, n_series)

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
            color_i = (
                color_opts[idx]
                if color_opts[idx] is not None
                else color_cycle[idx % len(color_cycle)]
            )
            line_i = line_styles[idx]
            marker_i = marker_styles[idx]
            markevery_i = marker_every_opts[idx]
            ci_level_i = ci_levels[idx]
            N = data_i.shape[0]
            max_samp = max_samples_opts[idx]
            count = min(N, max_samp) if max_samp is not None else N
            for j in range(count):
                ax.plot(
                    times_i,
                    data_i[j, :, dim_idx],
                    alpha=0.3,
                    lw=0.7,
                    linestyle=line_i or "-",
                    marker=marker_i,
                    markevery=markevery_i,
                )

            if plot_mean_opts[idx]:
                mean_i = data_i.mean(axis=0)
                ax.plot(
                    times_i,
                    mean_i[:, dim_idx],
                    color=color_i,
                    lw=2,
                    linestyle=line_i or "-",
                    marker=marker_i,
                    markevery=markevery_i,
                    label=f"{label}: sample mean",
                )
                if plot_ci_opts[idx]:
                    std_i = data_i.std(axis=0, ddof=1)
                    ax.fill_between(
                        times_i,
                        mean_i[:, dim_idx] - ci_level_i * std_i[:, dim_idx],
                        mean_i[:, dim_idx] + ci_level_i * std_i[:, dim_idx],
                        color=color_i,
                        alpha=0.2,
                        label=f"{label}: sample CI",
                    )

            if plot_median_opts[idx]:
                median_i = np.median(data_i, axis=0)
                ax.plot(
                    times_i,
                    median_i[:, dim_idx],
                    lw=2,
                    color=color_i,
                    linestyle=line_i or "--",
                    marker=marker_i,
                    markevery=markevery_i,
                    label=f"{label}: sample median",
                )

            tm_i = get_true_stat(true_mean, idx)
            ts_i = get_true_stat(true_std, idx)

            if tm_i is not None:
                ax.plot(
                    times_i,
                    tm_i[:, dim_idx],
                    color=color_i,
                    lw=2,
                    linestyle=line_i or "-",
                    marker=marker_i,
                    markevery=markevery_i,
                    label=f"{label}: true mean",
                )
                if ts_i is not None:
                    ax.fill_between(
                        times_i,
                        tm_i[:, dim_idx] - ci_level_i * ts_i[:, dim_idx],
                        tm_i[:, dim_idx] + ci_level_i * ts_i[:, dim_idx],
                        color=color_i,
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
    colors=None,
    line_style="-",
    marker_style=None,
    marker_every=None,
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
    max_samples : int or list
        Number of sample trajectories per dataset to plot.
    plot_mean/median : bool or list
        Draw sample mean/median trajectory.
    plot_ci : bool or list
        Draw ellipsoidal confidence region around mean if true_cov provided.
    true_mean, true_cov : arrays
        For reference (optional). true_cov used only if plot_ci and len(proj_dims)==2.
    colors/line_style/marker_style/marker_every : single value or list per series
    """
    #dims = np.atleast_1d(proj_dims)
    dims = [int(d) for d in np.atleast_1d(proj_dims)]

    if len(dims) not in (2, 3):
        raise ValueError("proj_dims must be pair or triple of indices.")

    if labels is None:
        labels = [f"series_{i}" for i in range(len(series_list))]

    n_series = len(series_list)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_opts = _broadcast_param(colors, n_series)
    line_styles = _broadcast_param(line_style, n_series)
    marker_styles = _broadcast_param(marker_style, n_series)
    marker_every_opts = _broadcast_param(marker_every, n_series)
    plot_mean_opts = _broadcast_param(plot_mean, n_series)
    plot_median_opts = _broadcast_param(plot_median, n_series)
    plot_ci_opts = _broadcast_param(plot_ci, n_series)
    ci_levels = _broadcast_param(ci_level, n_series)
    max_samples_opts = _broadcast_param(max_samples, n_series)

    fig = plt.figure(figsize=figsize)
    if len(dims) == 2:
        ax = fig.add_subplot(111)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # lazily import
        ax = fig.add_subplot(111, projection="3d")

    for idx, (data, label) in enumerate(zip(series_list, labels)):
        color_i = (
            color_opts[idx]
            if color_opts[idx] is not None
            else color_cycle[idx % len(color_cycle)]
        )
        data = np.asarray(data)
        N, T, d = data.shape
        max_samp = max_samples_opts[idx]
        count = min(N, max_samp) if max_samp is not None else N
        for i in range(count):
            if len(dims) == 2:
                ax.plot(
                    data[i, :, dims[0]],
                    data[i, :, dims[1]],
                    alpha=0.4,
                    color=color_i,
                    linestyle=line_styles[idx] or "-",
                    marker=marker_styles[idx],
                    markevery=marker_every_opts[idx],
                    label=label if i == 0 else "",
                )
            else:  # 3D
                ax.plot(
                    data[i, :, dims[0]],
                    data[i, :, dims[1]],
                    data[i, :, dims[2]],
                    alpha=0.3,
                    color=color_i,
                    linestyle=line_styles[idx] or "-",
                    marker=marker_styles[idx],
                )

        if plot_mean_opts[idx]:
            mean = data.mean(axis=0)[:, dims]
            if len(dims) == 2:
                ax.plot(
                    mean[:, 0],
                    mean[:, 1],
                    color=color_i,
                    lw=2,
                    linestyle=line_styles[idx] or "-",
                    marker=marker_styles[idx],
                    markevery=marker_every_opts[idx],
                    label=f"{label} mean",
                )
            else:
                ax.plot(
                    mean[:, 0],
                    mean[:, 1],
                    mean[:, 2],
                    color=color_i,
                    lw=2,
                    linestyle=line_styles[idx] or "-",
                    marker=marker_styles[idx],
                )

        if plot_median_opts[idx]:
            med = np.median(data, axis=0)[:, dims]
            if len(dims) == 2:
                ax.plot(
                    med[:, 0],
                    med[:, 1],
                    color=color_i,
                    lw=2,
                    linestyle=line_styles[idx] or "--",
                    marker=marker_styles[idx],
                    markevery=marker_every_opts[idx],
                    label=f"{label} median",
                )

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


def _prepare_sample_pairs(record, dims, sample_plot_count):
    pairs = []
    labels = []
    if sample_plot_count and "cfm_samples" in record:
        cfm_samples = np.asarray(record["cfm_samples"])[:, :, dims]
        pairs.append((record["future_times"], cfm_samples))
        labels.append("CFM samples")
    if sample_plot_count and "gp_samples" in record:
        gp_samples = np.asarray(record["gp_samples"])[:, :, dims]
        pairs.append((record["future_times"], gp_samples))
        labels.append("GP samples")
    return pairs, labels


def plot_block_weight_heatmaps(
    model,
    mode="block",
    include_shared=False,
    cmap="magma",
    annotate=False,
    norm="fro",
    title_prefix=None,
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

    title_prefix = title_prefix or getattr(model, "name", None) or model.__class__.__name__

    if mode == "block":
        block_vals = []
        block_masks = []
        for layer, layout in zip(layers, layouts):
            out_blocks = list(layout["out_blocks"])
            main_blocks = list(layout["in_blocks"])
            shared_blocks = list(layout["shared_block"]) if include_shared else []
            block_cols = main_blocks + shared_blocks

            arr = np.zeros((len(out_blocks), len(block_cols)), dtype=np.float32)
            mask_arr = np.zeros_like(arr, dtype=bool)

            weight = _get_layer_weight_tensor(layer)

            row_start = 0
            for r, out_dim in enumerate(out_blocks):
                col_start = 0
                for c, in_dim in enumerate(block_cols):
                    block = weight[row_start:row_start + out_dim, col_start:col_start + in_dim]
                    if block.numel() > 0:
                        if norm == "fro":
                            val = torch.linalg.norm(block, ord="fro").item()
                        elif norm == "absmax":
                            val = block.abs().max().item()
                        elif norm == "l1":
                            val = block.abs().sum().item()
                        else:
                            raise ValueError(f"Unsupported norm '{norm}'.")
                        arr[r, c] = val
                    if not _is_block_lower_tri_layer(layer):
                        mask_arr[r, c] = True

                    col_start += in_dim
                row_start += out_dim

            if _is_block_lower_tri_layer(layer):
                for i, j in layer._block_meta:
                    if not include_shared and j >= len(main_blocks):
                        continue
                    if j < mask_arr.shape[1]:
                        mask_arr[i, j] = True

            block_vals.append(arr)
            block_masks.append(mask_arr)
        cmap_obj = plt.get_cmap(cmap) if cmap is not None else _blue_red_positive_cmap()
        cmap_obj.set_bad(color="black")
        for idx, (ax, arr, mask_arr) in enumerate(zip(axes[0], block_vals, block_masks)):
            data_mask = mask_arr == 1
            if np.any(data_mask):
                data_min = arr[data_mask].min()
                data_max = arr[data_mask].max()
            else:
                data_min, data_max = 0.0, 1.0
            display = arr.copy()
            display[mask_arr == 0] = np.nan
            im = ax.imshow(display, cmap=cmap_obj, vmin=data_min, vmax=data_max if data_max > data_min else data_min + 1e-6, aspect="equal")
            ax.set_title(f"Layer {idx + 1}")
            ax.set_xlabel("Input blocks")
            ax.set_ylabel("Output blocks")
            if annotate:
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar_ticks = np.linspace(data_min, data_max if data_max > data_min else data_min + 1e-6, num=5)
            cbar.set_ticks(cbar_ticks)
        fig.suptitle(f"{title_prefix} - Block Heatmaps")
    elif mode == "full":
        processed = []
        for layer in layers:
            if _is_block_lower_tri_layer(layer):
                weight = layer.assemble_weight().detach().cpu().numpy()
                mask = np.zeros_like(weight)
                for block, (i, j) in zip(layer.blocks, layer._block_meta):
                    r0, r1 = layer.row_offsets[i], layer.row_offsets[i + 1]
                    c0, c1 = layer.col_offsets[j], layer.col_offsets[j + 1]
                    mask[r0:r1, c0:c1] = 1.0
            else:
                weight = layer.weight.detach().cpu().numpy()
                mask = layer.mask.detach().cpu().numpy()
            arr = np.abs(weight)
            processed.append((arr, mask))
        cmap_obj = plt.get_cmap(cmap) if cmap is not None else _blue_red_positive_cmap()
        cmap_obj.set_bad(color="black")
        for idx, (ax, (arr, mask)) in enumerate(zip(axes[0], processed)):
            data_mask = mask == 1
            if np.any(data_mask):
                data_min = arr[data_mask].min()
                data_max = arr[data_mask].max()
            else:
                data_min, data_max = 0.0, 1.0
            display = arr.copy()
            display[mask == 0] = np.nan
            im = ax.imshow(display, cmap=cmap_obj, vmin=data_min, vmax=data_max if data_max > data_min else data_min + 1e-6, aspect="equal")
            ax.set_title(f"Layer {idx + 1}")
            ax.set_xlabel("Input neurons")
            ax.set_ylabel("Output neurons")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar_ticks = np.linspace(data_min, data_max if data_max > data_min else data_min + 1e-6, num=5)
            cbar.set_ticks(cbar_ticks)
        fig.suptitle(f"{title_prefix} - Full Weight Heatmaps")
    else:
        raise ValueError("mode must be 'block' or 'full'")
    fig.tight_layout()
    return fig


def inspect_model_layers(model, include_block_stats=True, include_shared=False, eps=1e-9):
    """
    Print per-layer weight statistics to help diagnose degenerate parameters.

    Parameters
    ----------
    model : nn.Module
        Typically a MaskedBlockMLP exposing a .layers attribute.
    include_block_stats : bool
        When True and the model implements block_weight_magnitudes, also report
        per-block summaries.
    include_shared : bool
        Whether to include shared/time blocks when gathering block stats.
    eps : float
        Threshold for counting a value as numerically zero.

    Returns
    -------
    list of dict
        Raw statistics for each layer.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        raise TypeError("Model must expose .layers to inspect weights.")

    block_data = None
    if include_block_stats and hasattr(model, "block_weight_magnitudes"):
        try:
            block_data = model.block_weight_magnitudes(norm="fro", include_shared=include_shared)
        except Exception:
            block_data = None

    summary = []
    print("Layer weight statistics")
    print("-" * 80)
    for idx, layer in enumerate(layers):
        weight = _get_layer_weight_tensor(layer)
        tensor = weight.detach().float()
        numel = tensor.numel()
        layer_stats = {
            "layer_index": idx,
            "class": layer.__class__.__name__,
            "shape": tuple(tensor.shape),
            "numel": numel,
            "min": float(tensor.min().item()) if numel else 0.0,
            "max": float(tensor.max().item()) if numel else 0.0,
            "mean": float(tensor.mean().item()) if numel else 0.0,
            "std": float(tensor.std(unbiased=False).item()) if numel else 0.0,
            "fro_norm": float(torch.linalg.norm(tensor, ord="fro").item()) if numel else 0.0,
        }
        if numel:
            zeros = (tensor.abs() <= eps).sum().item()
            layer_stats["zero_fraction"] = zeros / numel
        if hasattr(layer, "mask"):
            mask = layer.mask.detach()
            layer_stats["active_connections"] = int(mask.sum().item())
            layer_stats["total_connections"] = mask.numel()
        summary.append(layer_stats)

        print(
            f"[Layer {idx}] {layer_stats['class']:>20} | shape={layer_stats['shape']}, "
            f"fro={layer_stats['fro_norm']:.3e}, mean={layer_stats['mean']:.3e}, "
            f"std={layer_stats['std']:.3e}, zero_frac={layer_stats.get('zero_fraction', 0.0):.2%}"
        )
        if "active_connections" in layer_stats:
            active = layer_stats["active_connections"]
            total = layer_stats["total_connections"]
            print(f"    Mask active {active}/{total} ({active/total:.2%})")

        if block_data is not None and idx < len(block_data):
            block_arr = block_data[idx]
            block_max = float(block_arr.max()) if block_arr.size else 0.0
            block_mean = float(block_arr.mean()) if block_arr.size else 0.0
            print(f"    Block magnitudes: mean={block_mean:.3e}, max={block_max:.3e}")

        bias = _get_layer_bias_tensor(layer)
        if bias is not None and bias.numel():
            b = bias.detach().float()
            print(
                f"    Bias: shape={tuple(b.shape)}, "
                f"min={float(b.min()):.3e}, max={float(b.max()):.3e}, "
                f"mean={float(b.mean()):.3e}, std={float(b.std(unbiased=False)):.3e}"
            )
    print("-" * 80)
    return summary


def plot_recursive_predictions(
    records: Sequence[Dict],
    dims: Optional[Sequence[int]] = None,
    max_steps: Optional[int] = None,
    connect_segments: bool = True,
    sample_plot_count: int = 0,
    marker_style: Optional[str] = None,
    marker_every: Optional[int] = None,
    past_line_style: str = "-",
    future_line_style: str = "-",
    cfm_line_style: str = "--",
    gp_line_style: str = "--",
    past_marker: Optional[str] = None,
    future_marker: Optional[str] = None,
    cfm_marker: Optional[str] = None,
    gp_marker: Optional[str] = None,
    title_prefix: str = "Trajectory",
    sample_colors=None,
    sample_line_styles=None,
    sample_marker_styles=None,
    sample_marker_every=None,
    sample_plot_mean=False,
    sample_plot_median=False,
    sample_plot_ci=False,
    sample_ci_level=2.0,
    sample_max_samples=None,
):
    """
    Plot recursive prediction records, optionally connecting past and future
    segments and overlaying sample trajectories from the CFM and GP models.

    Parameters
    ----------
    records : list of dicts
        Output from RecursiveCFMPredictor evaluation loop.
    dims : list of ints, optional
        Dimensions to plot. Defaults to all.
    max_steps : int, optional
        Plot up to this many records.
    connect_segments : bool
        When True, prepend the last observed point to the future curves.
    sample_plot_count : int
        If >0, include sample trajectories via ``plot_time_series``.
    *_line_style / *_marker : str or sequence
        Styling for the past, true future, CFM mean, and GP mean lines.
    sample_* parameters :
        Forwarded to ``plot_time_series``; each can be a single value applied to
        all series or a list providing per-series customization.
    """
    if not records:
        raise ValueError("`records` must contain at least one entry.")

    example = records[0]
    total_dims = example["future_truth"].shape[-1]
    dims = list(dims) if dims is not None else list(range(total_dims))
    steps = records if max_steps is None else records[:max_steps]

    for rec in steps:
        pairs, labels = _prepare_sample_pairs(rec, dims, sample_plot_count)
        fig = None
        axes = None
        if pairs:
            max_samples_arg = (
                sample_max_samples
                if sample_max_samples is not None
                else sample_plot_count
            )
            line_arg = (
                sample_line_styles if sample_line_styles is not None else future_line_style
            )
            marker_arg = (
                sample_marker_styles if sample_marker_styles is not None else marker_style
            )
            marker_every_arg = (
                sample_marker_every if sample_marker_every is not None else marker_every
            )
            fig = plot_time_series(
                pairs,
                labels=labels,
                dims=list(range(len(dims))),
                max_samples=max_samples_arg,
                separate_dims=True,
                figsize=(9, 3 * len(dims)),
                colors=sample_colors,
                line_style=line_arg,
                marker_style=marker_arg,
                marker_every=marker_every_arg,
                plot_mean=sample_plot_mean,
                plot_median=sample_plot_median,
                plot_ci=sample_plot_ci,
                ci_level=sample_ci_level,
            )
            axes = fig.axes
            for ax, dim_idx in zip(axes, dims):
                ax.set_ylabel(f"x_{dim_idx}")
        else:
            fig, axes = plt.subplots(
                len(dims), 1, sharex=True, figsize=(9, 3 * len(dims))
            )
            axes = np.atleast_1d(axes)

        past_times = np.asarray(rec.get("past_times", []))
        past_values = np.asarray(rec.get("past_values", []))
        future_times = np.asarray(rec["future_times"])
        last_time = past_times[-1] if past_times.size else None

        for ax, d_idx in zip(axes, dims):
            if past_times.size:
                ax.plot(
                    past_times,
                    past_values[:, d_idx],
                    color="k",
                    linewidth=2,
                    linestyle=past_line_style,
                    marker=past_marker,
                    markevery=marker_every,
                    label="Observed past",
                )

            truth_t, truth_vals = _connect_series(
                future_times,
                rec["future_truth"][:, d_idx],
                last_time,
                past_values[-1, d_idx] if past_times.size else None,
                connect_segments,
            )
            ax.plot(
                truth_t,
                truth_vals,
                color="C0",
                linewidth=2,
                linestyle=future_line_style,
                marker=future_marker,
                markevery=marker_every,
                label="True future",
            )

            cfm_t, cfm_vals = _connect_series(
                future_times,
                rec["cfm_mean"][:, d_idx],
                last_time,
                past_values[-1, d_idx] if past_times.size else None,
                connect_segments,
            )
            ax.plot(
                cfm_t,
                cfm_vals,
                color="C1",
                linewidth=2,
                linestyle=cfm_line_style,
                marker=cfm_marker,
                markevery=marker_every,
                label="CFM mean",
            )
            cfm_std = rec["cfm_std"][:, d_idx]
            ax.fill_between(
                future_times,
                rec["cfm_mean"][:, d_idx] - 2 * cfm_std,
                rec["cfm_mean"][:, d_idx] + 2 * cfm_std,
                color="C1",
                alpha=0.2,
                label="CFM 95% CI",
            )

            gp_t, gp_vals = _connect_series(
                future_times,
                rec["gp_mean"][:, d_idx],
                last_time,
                past_values[-1, d_idx] if past_times.size else None,
                connect_segments,
            )
            ax.plot(
                gp_t,
                gp_vals,
                color="C2",
                linewidth=2,
                linestyle=gp_line_style,
                marker=gp_marker,
                markevery=marker_every,
                label="GP mean",
            )
            gp_std = rec["gp_std"][:, d_idx]
            ax.fill_between(
                future_times,
                rec["gp_mean"][:, d_idx] - 2 * gp_std,
                rec["gp_mean"][:, d_idx] + 2 * gp_std,
                color="C2",
                alpha=0.2,
                label="GP 95% CI",
            )

            ax.set_ylabel(f"x_{d_idx}")
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(loc="upper right")
        step = rec.get("step", 0)
        traj = rec.get("trajectory_idx", 0)
        fig.suptitle(f"{title_prefix} {traj} – prediction step {step}")
        fig.tight_layout()
        plt.show()




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
