"""
Standalone training example showcasing the MaskedCFM workflow using the new
sampler + model bundle abstractions.
"""
import torch
from torch import nn
from torchdyn.core import NeuralODE
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from MaskedCFM.models import MaskedBlockMLP, CFMModelBundle
from MaskedCFM.random_processes.gp_registry import build_mean_fn, build_kernel
from MaskedCFM.random_processes.gp import GPSampler, GaussianVectorSampler


# --------------------------------------------------------------------------- #
# Conditional path / vector field utilities
# --------------------------------------------------------------------------- #
def SampleConditionalNoisyStraightPath(x0, x1, t, sigma=0.01):
    """Linear interpolation with Gaussian noise."""
    t = t.view(-1, 1)
    noise = sigma * torch.randn_like(x0)
    return (1.0 - t) * x0 + t * x1 + noise


def ConditionalVelocityField(x0, x1, t):
    """Analytic velocity for the straight-line homotopy."""
    _ = t  # unused but kept for API compatibility
    return x1 - x0


# def torch_wrapper(model: nn.Module):
#     """NeuralODE expects a module returning dx/dt given x."""
#     class VectorField(nn.Module):
#         def __init__(self, base):
#             super().__init__()
#             self.base = base

#         def forward(self, t, x):
#             t_column = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
#             inp = torch.cat([x, t_column], dim=-1)
#             return self.base(inp)

#     return VectorField(model)


def make_node(model):
    return NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)


class DeviceSampler:
    """Wrap a sampler to force outputs onto a specific device."""

    def __init__(self, sampler, device):
        self.sampler = sampler
        self.device = device

    def sample(self, batch_size, flatten=True):
        out = self.sampler.sample(batch_size, flatten=flatten)
        return out.to(self.device)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate_forward(bundle, label, batch_size, target_mean, T):
    with torch.no_grad():
        preds = bundle.forward_map(batch_size, flatten=True)
    target_mean = target_mean.to(preds.device)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    diff_mean = mean_pred - target_mean
    mean_err = torch.sqrt(torch.sum(diff_mean ** 2) / T).item()
    diff_samples = preds - target_mean
    sample_err = torch.sqrt(torch.mean(torch.sum(diff_samples ** 2, dim=1) / T)).item()
    return {
        "label": label,
        "mean_pred": mean_pred.cpu(),
        "std_pred": std_pred.cpu(),
        "mean_diff_target": mean_err,
        "mean_dev_target": sample_err,
    }


def evaluate_backward(bundle, label, batch_size, prior_mean, T):
    with torch.no_grad():
        preds = bundle.backward_map(batch_size, flatten=True)
    prior_mean = prior_mean.to(preds.device)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    diff_mean = mean_pred - prior_mean
    mean_err = torch.sqrt(torch.sum(diff_mean ** 2) / T).item()
    diff_samples = preds - prior_mean
    sample_err = torch.sqrt(torch.mean(torch.sum(diff_samples ** 2, dim=1) / T)).item()
    return {
        "label": label,
        "mean_pred": mean_pred.cpu(),
        "std_pred": std_pred.cpu(),
        "mean_diff_prior": mean_err,
        "mean_dev_prior": sample_err,
    }


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    dim = 2
    T = 50
    times = torch.linspace(0.0, 1.0, T, device=device)

    mean_fn = build_mean_fn("spiral", {"freq": 1.0})
    cov_kernel = build_kernel("rbf", {"sigma": 1.0, "tau": 0.1})
    gp_sampler = GPSampler(mean_fn=mean_fn, cov_kernel=cov_kernel, times=times.cpu(), jitter=1e-4)
    gp_mean, _ = gp_sampler.get_stats(flatten=False)

    prior_sampler = GaussianVectorSampler(dim=dim, T=T, mean=None, cov=None, device=device)
    mean_prior, _ = prior_sampler.get_stats(flatten=True)
    target_sampler = DeviceSampler(gp_sampler, device=device)
    mean_target = gp_mean.reshape(-1).to(device)

    dim_w = 16
    num_layers = 3
    model_causal = MaskedBlockMLP(T=T, in_dim=dim, out_dim=dim, hidden_per_t=(dim_w,) * num_layers, causal=True, time_varying=True).to(device)
    model_full = MaskedBlockMLP(T=T, in_dim=dim, out_dim=dim, hidden_per_t=(dim_w,) * num_layers, causal=False, time_varying=True).to(device)

    bundle_causal = CFMModelBundle(
        model=model_causal,
        cond_path_fn=SampleConditionalNoisyStraightPath,
        cond_vec_field_fn=ConditionalVelocityField,
        prior_sampler=prior_sampler,
        target_sampler=target_sampler,
        ode_factory=make_node,
        device=device,
    )

    bundle_full = CFMModelBundle(
        model=model_full,
        cond_path_fn=SampleConditionalNoisyStraightPath,
        cond_vec_field_fn=ConditionalVelocityField,
        prior_sampler=prior_sampler,
        target_sampler=target_sampler,
        ode_factory=make_node,
        device=device,
    )

    optimizer_causal = torch.optim.Adam(model_causal.parameters(), lr=1e-3)
    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)

    batch_size = 2 ** 10
    batch_val = 2 ** 10
    num_steps = 2000
    eval_every = 200

    train_history = []
    val_history = []

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("Step {task.fields[step]:>4}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task(
            "Training",
            total=num_steps,
            step=0,
            loss_c=0.0,
            loss_f=0.0,
        )

        for step in range(num_steps):
            optimizer_causal.zero_grad()
            optimizer_full.zero_grad()

            shared = bundle_causal.draw_samples(batch_size)
            loss_causal = bundle_causal.loss(batch_size, shared_data=shared)
            loss_full = bundle_full.loss(batch_size, shared_data=shared)

            loss_causal.backward()
            loss_full.backward()
            optimizer_causal.step()
            optimizer_full.step()

            train_history.append({
                "step": step + 1,
                "loss_causal": float(loss_causal.detach()),
                "loss_full": float(loss_full.detach()),
            })

            fields = {
                "step": step + 1,
                "loss_c": loss_causal.item(),
                "loss_f": loss_full.item(),
            }

            if (step + 1) % eval_every == 0:
                forward_stats_causal = evaluate_forward(bundle_causal, "causal", batch_val, mean_target, T)
                forward_stats_full = evaluate_forward(bundle_full, "full", batch_val, mean_target, T)
                backward_stats_causal = evaluate_backward(bundle_causal, "causal", batch_val, mean_prior, T)
                backward_stats_full = evaluate_backward(bundle_full, "full", batch_val, mean_prior, T)

                val_history.append({
                    "step": step + 1,
                    "loss_causal": fields["loss_c"],
                    "loss_full": fields["loss_f"],
                    "forward_stats_causal": forward_stats_causal,
                    "forward_stats_full": forward_stats_full,
                    "backward_stats_causal": backward_stats_causal,
                    "backward_stats_full": backward_stats_full,
                })

            progress.update(task_id, advance=1, **fields)

    return train_history, val_history


if __name__ == "__main__":
    main()
