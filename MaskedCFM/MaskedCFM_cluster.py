# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
print("Importing libraries...")

import os
import time
import json

from io import BytesIO
from PIL import Image

from pathlib import Path
from datetime import datetime
from importlib import reload

import math
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader

import ot as pot
import torchdyn
from torchdyn.core import NeuralODE
from torchdyn.datasets import *

from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from torchcfm.optimal_transport import *

from tqdm import tqdm
from tqdm.auto import trange
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from random_processes import *
from models import *

# --------------------------------------------------------------------------- #
# Device configuration and seed initialization
# --------------------------------------------------------------------------- #
print("Configuring device and initializing seed...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed initialization
def init_seed(use_fixed=True, fixed_seed=42):
    if use_fixed:
        seed = fixed_seed
    else:
        seed = int.from_bytes(os.urandom(64), "little")
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

init_seed(use_fixed=True, fixed_seed=42)


# Saving directories
savedir = Path("savedir/MaskedCFM")
savedir.mkdir(parents=True, exist_ok=True)

def snapshot_state_dict(state):
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().clone().cpu()
        elif isinstance(v, dict):
            out[k] = snapshot_state_dict(v)
        else:
            out[k] = v
    return out

def prepare_run_folder(root: Path):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return stamp, run_dir

run_id, run_dir = prepare_run_folder(savedir)
(loss_plots_path, samples_fig_path) = (run_dir / "val_metrics.png", run_dir / "forward_samples.png")

def to_serializable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj



# --------------------------------------------------------------------------- #
# Conditional path / vector field utilities and Neural ODE solver factory
# --------------------------------------------------------------------------- #
print("Defining conditional path and vector field utilities...")

def SampleConditionalNoisyStraightPath(x0, x1, t, sigma=0.01):
    """Linear interpolation with Gaussian noise."""
    t = t.view(-1, 1)
    noise = sigma * torch.randn_like(x0)
    return (1.0 - t) * x0 + t * x1 + noise


def ConditionalVelocityField(x0, x1, t):
    """Analytic velocity for the straight-line homotopy."""
    _ = t  # unused but kept for API compatibility
    return x1 - x0


def SampleConditionalBrownianBridgePath(x0, x1, t, sigma=0.01):
    """
    Draw a sample from the probability path xt = (1 - t) * x0 + t * x1 + sigma * (Wt - t * W1)
    where Wt is a standard Brownian motion.
    """
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    sigma_t = sigma * torch.sqrt(t * (1 - t))  #FIX THIS, should be Brownian bridge stddev
    epsilon = torch.randn_like(x0)             #FIX THIS, should be Brownian bridge noise
    return mu_t + sigma_t * epsilon


def make_node(model : torch.nn.Module):
    return NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)



# --------------------------------------------------------------------------- #
# Prior and target distribution samplers
# --------------------------------------------------------------------------- #
print("Defining prior and target distribution samplers...")

# # Sampler from stochastically driven Lorenz attractor
# dim = 3
# T = 30
# t_min, t_max = 0.0, 0.5

# lorenz_drift, lorenz_diff = lorenz_system(sigma=10.0, rho=28.0, beta=8/3, noise_scale=0.5)

# def lorenz_init_sampler(train_batch_size):
#     return torch.empty(train_batch_size, 3).uniform_(-15, 15)

# lorenz_sampler = SDESampler(
#     drift_fn=lorenz_drift,
#     diffusion_fn=lorenz_diff,
#     t0=t_min,
#     t1=t_max,
#     steps=T-1,
#     init_sampler=lorenz_init_sampler,
#     method="euler",
# )


# Sampler from Gaussian process with RBF kernel
dim = 2
T = 30
t_min, t_max = 0.0, 0.5

mean_fn = build_mean_fn("spiral", {"freq": 5.0})
A = torch.tensor([[1.0, 0.3], [0.3, 0.8]])
cov_kernel = build_kernel("rbf", {"sigma": 1.0, "tau": 5.0, "Cov": A})

times = torch.linspace(t_min, t_max, T)
gp_sampler = GPSampler(mean_fn=mean_fn, cov_kernel=cov_kernel, times=times, ldl_decomp=True, jitter=1e-4)
gp_mean, gp_cov = gp_sampler.get_stats()


# prior: standard Gaussian
std_gaussian_sampler = GaussianVectorSampler(dim=dim, T=T)
std_gaussian_mean, std_gaussian_cov = std_gaussian_sampler.get_stats(flatten=False)

#var = 0.01 
mean_prior = std_gaussian_mean.reshape(-1).to(device)
mean_target = gp_mean.reshape(-1).to(device)

prior_sampler = std_gaussian_sampler
target_sampler = gp_sampler



# --------------------------------------------------------------------------- #
# Evaluation helpers (forward and backward model samplers)
# --------------------------------------------------------------------------- #
print("Defining evaluation helpers...")

def evaluate_forward(bundle, label, batch_size, target_mean, T):
    with torch.no_grad():
        preds = bundle.forward_map(batch_size=batch_size, flatten=True)
    true_target_samples = bundle._draw_target(batch_size)
    target_mean = target_mean.to(preds.device)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    diff_mean = mean_pred - target_mean
    mean_err = torch.sqrt(torch.sum(diff_mean ** 2) / T).item()
    diff_samples = preds - target_mean
    sample_err = torch.sqrt(torch.mean(torch.sum(diff_samples ** 2, dim=1) / T)).item()

    w2 = wasserstein(preds, true_target_samples, method="exact", power=2)
    w2_dist = math.sqrt((w2 ** 2) / T)
    w2_dist = float(w2_dist)  # ensure plain float
 
    return {
        "label": label,
        "mean_pred": mean_pred.cpu(),
        "std_pred": std_pred.cpu(),
        "mean_diff_target": mean_err,
        "mean_dev_target": sample_err,
        "w2_dist": w2_dist,
    }

def evaluate_backward(bundle, label, batch_size, prior_mean, T):
    with torch.no_grad():
        preds = bundle.backward_map(batch_size=batch_size, flatten=True)
    true_prior_samples = bundle._draw_prior(batch_size)
    prior_mean = prior_mean.to(preds.device)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    diff_mean = mean_pred - prior_mean
    mean_err = torch.sqrt(torch.sum(diff_mean ** 2) / T).item()
    diff_samples = preds - prior_mean
    sample_err = torch.sqrt(torch.mean(torch.sum(diff_samples ** 2, dim=1) / T)).item()

    w2 = wasserstein(preds, true_prior_samples, method="exact", power=2)
    w2_dist = math.sqrt((w2 ** 2) / T)
    w2_dist = float(w2_dist)  # ensure plain float
    
    return {
        "label": label,
        "mean_pred": mean_pred.cpu(),
        "std_pred": std_pred.cpu(),
        "mean_diff_prior": mean_err,
        "mean_dev_prior": sample_err,
        "w2_dist": w2_dist,
    }


# --------------------------------------------------------------------------- #
# Create model bundles to train and evaluate
# --------------------------------------------------------------------------- #
print("Creating model bundles...")

model_specs = []

dim_w = 4 #hidden layers dimension per T
num_layers = 3
lr_causal, lr_full = 1e-3, 1e-3


model_causal_indep= MaskedBlockMLP(T=T, in_dim=dim, out_dim=dim, hidden_per_t=(dim_w,) * num_layers, causal=True, time_varying=True, block_init="xavier", use_batch_norm=True)
optimizer_causal_indep = torch.optim.Adam(model_causal_indep.parameters(), lr=lr_causal)
bundle_causal_indep = CFMModelBundle(
    model=model_causal_indep,
    cond_path_fn=SampleConditionalNoisyStraightPath,
    cond_vec_field_fn=ConditionalVelocityField,
    prior_sampler=prior_sampler,
    target_sampler=target_sampler,
    ode_factory=make_node,
    loss_fn=nn.MSELoss(),
    optimizer=optimizer_causal_indep,
    device=device,
)
model_specs.append(build_bundle(name = "causal_indep", long_name = "Causal Block MLP, Indep. Coupling", bundle = bundle_causal_indep))

model_full_indep   = MaskedBlockMLP(T=T, in_dim=dim, out_dim=dim, hidden_per_t=(dim_w,) * num_layers, causal=False, time_varying=True, block_init="xavier", use_batch_norm=True)
optimizer_full_indep   = torch.optim.Adam(model_full_indep.parameters(), lr=lr_full)
bundle_full_indep = CFMModelBundle(
    model=model_full_indep,
    cond_path_fn=SampleConditionalNoisyStraightPath,
    cond_vec_field_fn=ConditionalVelocityField,
    prior_sampler=prior_sampler,
    target_sampler=target_sampler,
    ode_factory=make_node,
    loss_fn=nn.MSELoss(),
    optimizer=optimizer_full_indep,
    device=device,
)
model_specs.append(build_bundle(name = "full_indep", long_name = "Full Block MLP, Indep. Coupling", bundle = bundle_full_indep))



# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
print("Starting training loop...")

train_batch_size = 2**12
val_batch_size = 2**12
num_steps = 50000
eval_every = 500
prog_every = 25

console = Console(force_jupyter=True)  # ensures Rich draws in notebooks

def snapshot_state_dict(state_dict):
    return {k: v.clone().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}

with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("Step {task.fields[step]:>4}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        #refresh_per_second=1
    ) as progress:
        task_id = progress.add_task(
            "Training",
            total=num_steps,
            step=0,
        )

        for step in range(num_steps):
            train_shared_data = model_specs[0].bundle.draw_samples(train_batch_size)

            for spec in model_specs:
                spec.bundle.train(True)
                train_loss = spec.bundle.loss(train_batch_size, shared_data=train_shared_data)
                spec.train_history.append({
                    "step": step + 1,
                    "train_loss": float(train_loss.detach()),
                })

            if (step + 1) % eval_every == 0:
                #val_shared_data = model_specs[0].bundle.draw_samples(val_batch_size)

                for spec in model_specs:
                    spec.bundle.eval()
                    #val_loss = spec.bundle.loss(val_batch_size, shared_data=val_shared_data)
                    val_loss = spec.bundle.loss(val_batch_size)
                    f_stats = evaluate_forward(spec.bundle, spec.name, val_batch_size, mean_target, T)
                    b_stats = evaluate_backward(spec.bundle, spec.name, val_batch_size, mean_prior, T)
                    spec.val_history.append({
                        "step": step + 1,
                        "val_loss": float(val_loss.detach()),
                        "forward": f_stats,
                        "backward": b_stats,
                    })
                    spec.checkpoints.append({
                        "step": step + 1,
                        "model": snapshot_state_dict(spec.bundle.model.state_dict()),
                        "optimizer": snapshot_state_dict(spec.bundle.optimizer.state_dict()),
                    })
                    
            #if (step + 1) % prog_every == 0:
            progress.update(task_id, advance=1, step=step + 1)



# --------------------------------------------------------------------------- #
# Saving data
# --------------------------------------------------------------------------- #
print("Saving training state and metadata...")

def to_serializable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

run_state = {
    "run_id": run_id,
    "step": num_steps,
    "samplers": {
        "prior": {
            "class": prior_sampler.__class__.__name__,
            "module": prior_sampler.__class__.__module__,
            "state_dict": prior_sampler.state_dict(),
        },
        "target": {
            "class": target_sampler.__class__.__name__,
            "module": target_sampler.__class__.__module__,
            "state_dict": target_sampler.state_dict(),
        },
    },
    "bundles": serialize_model_specs(model_specs),
}

torch.save(run_state, run_dir / "training_state.pt")

payload = {
    "run_id": run_id,
    "timestamp": datetime.now().isoformat(),
    "training": {
        "num_steps": num_steps,
        "eval_every": eval_every,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
    },
    "figures": {
        spec.name: {
            "metrics": str(run_dir / f"{spec.name}_metrics.png"),
            "timeseries": str(run_dir / f"{spec.name}_timeseries.png"),
            "phase": str(run_dir / f"{spec.name}_phase.png"),
            "blocks": str(run_dir / f"{spec.name}_blocks_heatmap.png"),
            "weights": str(run_dir / f"{spec.name}_weights_heatmap.png"),
        }
        for spec in model_specs
    },
}

with open(run_dir / "run_meta.json", "w") as f:
    json.dump(to_serializable(payload), f, indent=2)



# --------------------------------------------------------------------------- #
# Plotting training and validation data
# --------------------------------------------------------------------------- #
print("Plotting training and validation metrics...")

fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
axes = axes.ravel()

for spec in model_specs:
    if not spec.val_history:
        continue
    steps = [v["step"] for v in spec.val_history]
    val_loss = [v["val_loss"] for v in spec.val_history]
    train_steps = [t["step"] for t in spec.train_history]
    train_loss = [t["train_loss"] for t in spec.train_history]

    f_mean = [v["forward"]["mean_diff_target"] for v in spec.val_history]
    f_dev  = [v["forward"]["mean_dev_target"] for v in spec.val_history]
    f_w2   = [v["forward"]["w2_dist"] for v in spec.val_history]
    b_mean = [v["backward"]["mean_diff_prior"] for v in spec.val_history]
    b_dev  = [v["backward"]["mean_dev_prior"] for v in spec.val_history]
    b_w2   = [v["backward"]["w2_dist"] for v in spec.val_history]

    axes[0].plot(train_steps, train_loss, label=spec.name)
    axes[1].plot(steps, val_loss, label=spec.name)
    axes[2].plot(steps, f_mean, label=spec.name)
    axes[3].plot(steps, b_mean, label=spec.name)
    axes[4].plot(steps, f_dev, label=spec.name)
    axes[5].plot(steps, b_dev, label=spec.name)
    axes[6].plot(steps, f_w2, label=spec.name)
    axes[7].plot(steps, b_w2, label=spec.name)

titles = [
    "Train Loss",
    "Validation Loss",
    "Forward Delta_Mean",
    "Backward Delta_Mean",
    "Forward Mean Dev",
    "Backward Mean Dev",
    "Forward Delta_W2",
    "Backward Delta_W2",
]
for ax, title in zip(axes, titles):
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("Step")
fig.tight_layout()
fig.savefig(run_dir / "all_specs_metrics.png", dpi=150)
# plt.show()



# --------------------------------------------------------------------------- #
# Plotting generated time series
# --------------------------------------------------------------------------- #
print("Plotting generated time series...")

eval_batch = 2**8
time_grid = gp_sampler.times.cpu().numpy()
gp_mean_flat, gp_cov = gp_sampler.get_stats(flatten=True)
gp_mean = gp_mean_flat.reshape(T, dim).cpu().numpy()
gp_std = torch.sqrt(torch.diagonal(gp_cov).reshape(T, dim)).cpu().numpy()

x0_shared = prior_sampler.sample(eval_batch).to(device)
x1_true = target_sampler.sample(eval_batch).to(device).reshape(eval_batch, T, dim).cpu().numpy()

bundle_predictions = {}
for spec in model_specs:
    preds = spec.bundle.forward_map(inputs=x0_shared, flatten=True)
    data_pred = preds.detach().cpu().view(eval_batch, T, dim).numpy()
    bundle_predictions[spec.name] = data_pred

# 1) individual figures per bundle
for spec in model_specs:
    data_pred = bundle_predictions[spec.name]

    fig_ts = plot_time_series(
        [(time_grid, data_pred), (time_grid, x1_true)],
        labels=[f"{spec.name} forward", "True Target"],
        plot_mean=True,
        plot_ci=True,
        ci_level=2.0,
        max_samples=0,
        separate_dims=True,
        figsize=(10, 4),
    )
    fig_ts.savefig(run_dir / f"{spec.name}_timeseries.png", dpi=150)
    # plt.show()

    fig_phase = plot_state_space(
        [data_pred, x1_true],
        proj_dims=(0, 1),
        labels=[spec.name, "True Target"],
        max_samples=0,
        plot_mean=True,
    )
    fig_phase.savefig(run_dir / f"{spec.name}_phase.png", dpi=150)
    # plt.show()

# 2) overlay figures for a chosen subset
overlay_model_names = ["causal_indep", "full_indep"]
overlay_specs = [spec for spec in model_specs if spec.name in overlay_model_names]
overlay_data = [(time_grid, bundle_predictions[spec.name]) for spec in overlay_specs]
overlay_labels = [spec.name for spec in overlay_specs]

fig_ts_overlay = plot_time_series(
    overlay_data + [(time_grid, x1_true)],
    labels=overlay_labels + ["True Target"],
    plot_mean=True,
    plot_ci=False,
    ci_level=2.0,
    max_samples=0,
    separate_dims=True,
    figsize=(10, 4),
)
fig_ts_overlay.savefig(run_dir / "overlay_timeseries.png", dpi=150)
# plt.show()

fig_phase_overlay = plot_state_space(
    [bundle_predictions[spec.name] for spec in overlay_specs] + [x1_true],
    proj_dims=(0, 1),
    labels=overlay_labels + ["True Target"],
    max_samples=0,
    plot_mean=True,
)
fig_phase_overlay.savefig(run_dir / "overlay_phase.png", dpi=150)
# plt.show()



# --------------------------------------------------------------------------- #
# Plotting weight heatmaps of selected models
# --------------------------------------------------------------------------- #
print("Plotting weight heatmaps of selected models...")

selected_model_names = ["causal_indep", "full_indep"]  # change to whichever bundles you want
selected_specs = [spec for spec in model_specs if spec.name in selected_model_names]

for spec in selected_specs:
    model = spec.bundle.model
    fig_block = plot_block_weight_heatmaps(
        model,
        mode="block",
        include_shared=False,
        cmap="viridis",
        annotate=False,
        norm="fro",
        title_prefix=spec.name,
    )
    fig_block.savefig(run_dir / f"{spec.name}_blocks_heatmap.png", dpi=150)
    # plt.show()

    fig_full = plot_block_weight_heatmaps(
        model,
        mode="full",
        include_shared=False,
        cmap="viridis",
        annotate=False,
        norm="fro",
        title_prefix=spec.name,
    )
    fig_full.savefig(run_dir / f"{spec.name}_weights_heatmap.png", dpi=150)
    # plt.show()



# --------------------------------------------------------------------------- #
# Recursive prediction evaluation with CFM model
# --------------------------------------------------------------------------- #
print("Starting recursive prediction evaluation with CFM model...")

bundle_causal = model_specs[0].bundle

cfm_predictor = RecursiveCFMPredictor(
    bundle=bundle_causal,
    prior_sampler=prior_sampler,
    T=T,
    block_dim=dim,
    device=device,
)

# --- Recursive evaluation -----------------------------------------------------
num_conditional_samples = 2**5
num_obs_samples = 1
plot_sample_count = 5
selected_traj_idx = 0
all_step_mse = []
all_step_w2 = []
selected_records = []

gp_samples = gp_sampler.sample(batch_size=num_obs_samples, flatten=False).to(device)  # (N, T, dim)

for traj_idx in range(gp_samples.shape[0]):
    cfm_predictor.reset()
    gp_cond = gp_sampler.recursive_conditioner(num_trajectories=1)
    traj = gp_samples[traj_idx]

    for step in range(T):
        # print(f"Traj idx: {traj_idx}/{num_obs_samples}, Step: {step}/{T}")
        if step > 0:
            obs = traj[step - 1:step].unsqueeze(0)  # (1, 1, dim)
            gp_cond.observe(obs)
            cfm_predictor.observe(step - 1, traj[step - 1])

        dist = gp_cond.distribution()
        if dist.future_blocks == 0:
            break

        target_mean = dist.mean(flatten=False).squeeze(0)
        target_samples = dist.sample(num_conditional_samples, flatten=False)

        cfm_samples, cfm_mean, cfm_std = cfm_predictor.predict_next(num_conditional_samples)
        cfm_future = cfm_samples[:, step:, :]  # (num_samples, future_blocks, dim)

        true_future = traj[step:]
        mse = torch.mean((target_mean - true_future) ** 2).item()
        all_step_mse.append(mse)

        cfm_flat = cfm_future.reshape(num_conditional_samples, -1)
        target_flat = target_samples.reshape(num_conditional_samples, -1)
        w2 = wasserstein(cfm_flat, target_flat, method="exact", power=2)
        all_step_w2.append(float(w2))

        if traj_idx == selected_traj_idx:
            future_std = torch.sqrt(torch.diagonal(dist.covariance).view(dist.future_blocks, dim))
            selected_records.append({
                "step": step,
                "past_times": times[:step].cpu().numpy(),
                "past_values": traj[:step].cpu().numpy(),
                "future_times": times[step:].cpu().numpy(),
                "future_truth": true_future.cpu().numpy(),
                "cfm_mean": cfm_mean[step:].detach().cpu().numpy(),
                "cfm_std": cfm_std[step:].detach().cpu().numpy(),
                "gp_mean": target_mean.cpu().numpy(),
                "gp_std": future_std.cpu().numpy(),
                "cfm_samples": cfm_future[:plot_sample_count].detach().cpu().numpy(),
                "gp_samples": target_samples[:plot_sample_count].detach().cpu().numpy(),
            })

avg_mse = float(np.mean(all_step_mse))
avg_w2 = float(np.mean(all_step_w2))
print(f"Average future MSE: {avg_mse:0.6f}")
print(f"Average W2 distance: {avg_w2:0.6f}")



# --------------------------------------------------------------------------- #
# Plotting recursive prediction results
# --------------------------------------------------------------------------- #
print("Plotting recursive prediction results...")

prediction_figs = plot_recursive_predictions(selected_records, dims=[0, 1], max_steps=T-1)

frames = []
for fig in prediction_figs:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
    plt.close(fig)  # optional: free memory

if frames:
    gif_path = run_dir / f"{spec.name}_recursive_predictions.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,   # ms per frame
        loop=0          # 0=loop forever
    )

# plt.show()


