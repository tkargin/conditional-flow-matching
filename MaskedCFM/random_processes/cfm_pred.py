import torch
from typing import Dict, Optional

__all__ = ["RecursiveCFMPredictor"]


class RecursiveCFMPredictor:
    """
    Recursive predictor for causal CFM models.

    Tracks observed blocks and reuses previously solved initial
    conditions when new observations arrive.
    """

    def __init__(
        self,
        bundle,
        prior_sampler,
        T: int,
        block_dim: int,
        device: Optional[torch.device] = None,
    ):
        self.bundle = bundle
        self.prior_sampler = prior_sampler
        self.T = int(T)
        self.block_dim = int(block_dim)
        self.total_dim = self.T * self.block_dim
        self.device = device or bundle.device

        self.observed: Dict[int, torch.Tensor] = {}
        self.cached_initials: Dict[int, torch.Tensor] = {}
        self.prefix_initial_vector: Optional[torch.Tensor] = None

    def reset(self):
        self.observed.clear()
        self.cached_initials.clear()
        self.prefix_initial_vector = None

    def observe(self, block_idx: int, value: torch.Tensor):
        """Register a new observation y_k at block index (0-based)."""
        block_idx = int(block_idx)
        value = value.to(self.device).detach()
        self.observed[block_idx] = value.clone()
        self._compute_initial_for_block(block_idx)
        self._update_prefix_vector()

    def _update_prefix_vector(self):
        if not self.cached_initials:
            self.prefix_initial_vector = None
            return
        vec = torch.zeros(1, self.total_dim, device=self.device)
        for idx, init in self.cached_initials.items():
            start = idx * self.block_dim
            end = start + self.block_dim
            vec[:, start:end] = init
        self.prefix_initial_vector = vec

    def _compute_initial_for_block(self, idx: int):
        """Compute initial x_idx(0) for a newly observed block."""
        state_T = torch.zeros(1, self.total_dim, device=self.device)
        for j, val in self.observed.items():
            if j <= idx:
                start = j * self.block_dim
                end = start + self.block_dim
                state_T[:, start:end] = val

        ode = self.bundle._require_ode()
        with torch.no_grad():
            traj = ode.trajectory(
                state_T,
                t_span=torch.linspace(1.0, 0.0, 2, device=self.device),
            )
        x0 = traj[-1]
        start = idx * self.block_dim
        end = start + self.block_dim
        self.cached_initials[idx] = x0[:, start:end].detach()

    def sample_conditional(self, num_samples: int, flatten: bool = True):
        """Draw samples from p(x | y_{1:k})."""
        num_samples = int(num_samples)
        if not self.observed:
            preds = self.bundle.forward_map(batch_size=num_samples, flatten=flatten)
            if not flatten:
                preds = preds.view(num_samples, self.T, self.block_dim)
            return preds
        x0 = torch.zeros(num_samples, self.total_dim, device=self.device)
        if self.prefix_initial_vector is not None:
            prefix = self.prefix_initial_vector.expand(num_samples, -1)
            x0[:, : prefix.shape[1]] = prefix
        # fill observed blocks with cached initials
        for idx, init_block in self.cached_initials.items():
            start = idx * self.block_dim
            end = start + self.block_dim
            x0[:, start:end] = init_block.expand(num_samples, -1)

        # sample unobserved blocks from prior at t=0
        if len(self.observed) < self.T:
            prior_samples = self.prior_sampler.sample(num_samples, flatten=False)
            prior_samples = prior_samples.to(self.device)
            for idx in range(self.T):
                if idx in self.observed:
                    continue
                start = idx * self.block_dim
                end = start + self.block_dim
                x0[:, start:end] = prior_samples[:, idx, :]

        preds = self.bundle.forward_map(inputs=x0, flatten=flatten)
        if flatten:
            return preds
        else:
            return preds.view(num_samples, self.T, self.block_dim)

    def predict_next(self, num_samples: int):
        """Convenience wrapper that returns samples and their mean/std."""
        samples = self.sample_conditional(num_samples, flatten=False)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return samples, mean, std
