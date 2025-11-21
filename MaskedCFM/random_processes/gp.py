import math
import numpy as np
import torch
import torch.nn as nn
from fbm import FBM

from .base_sampler import Sampler 
from .gp_registry  import build_kernel, build_mean_fn


##########################
# GP Samplers Class
##########################

class GaussianVectorSampler(Sampler):
    def __init__(self, dim, T, mean=None, cov=None, device="cpu"):
        super().__init__("GaussianVectorSampler", dim=dim, T=T)
        self.dim = int(dim)
        self.T = int(T)
        self.device = torch.device(device)

        self.mean = self._prepare_mean(mean)
        self.cov = self._prepare_cov(cov)
        self.mvn = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.cov)

    def _prepare_mean(self, mean):
        total = self.dim * self.T
        if mean is None:
            return torch.zeros(total, device=self.device)
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        if mean_tensor.dim() == 0:
            return mean_tensor.expand(total)
        if mean_tensor.numel() == self.dim:
            mean_tensor = mean_tensor.reshape(1, self.dim).expand(self.T, -1)
        elif mean_tensor.numel() == total:
            mean_tensor = mean_tensor.reshape(self.T, self.dim)
        else:
            raise ValueError("Mean must be None, scalar, shape (dim,), (T, dim), or length T*dim.")
        return mean_tensor.reshape(-1)

    def _prepare_cov(self, cov):
        total = self.dim * self.T
        if cov is None:
            return torch.eye(total, device=self.device)
        cov_tensor = torch.as_tensor(cov, dtype=torch.float32, device=self.device)
        if cov_tensor.dim() == 0:
            return cov_tensor * torch.eye(total, device=self.device)
        if cov_tensor.shape == (self.dim, self.dim):
            eye_T = torch.eye(self.T, device=self.device)
            return torch.kron(eye_T, cov_tensor)
        if cov_tensor.shape == (total, total):
            return cov_tensor
        raise ValueError("Covariance must be None, scalar, shape (dim, dim), or (T*dim, T*dim).")

    def sample(self, batch_size, flatten=True):
        samples = self.mvn.sample((batch_size,))
        if not flatten:
            samples = samples.view(batch_size, self.T, self.dim)
        return samples

    def get_stats(self, flatten=False):
        mean = self.mean if flatten else self.mean.view(self.T, self.dim)
        return mean, self.cov

    def state_dict(self):
        state = super().state_dict()
        state["config"] = dict(state["config"])
        state["config"]["mean"] = self.mean.cpu().tolist()
        state["config"]["cov"] = self.cov.cpu().tolist()
        return state

    @classmethod
    def from_state_dict(cls, state):
        cfg = state["config"]
        mean = torch.tensor(cfg.get("mean"))
        cov = torch.tensor(cfg.get("cov"))
        return cls(
            dim=cfg["dim"],
            T=cfg["T"],
            mean=mean,
            cov=cov,
        )


class GPSampler(Sampler):
    """
    Efficient GP sampler that caches mean, covariance, and block-LDL factors
    for a given set of sampling times. You can update the time grid later
    without rebuilding the entire class.
    """

    def __init__(
        self,
        mean_fn,
        cov_kernel,
        times,
        jitter=1e-6,
        ldl_decomp=True,
        mean_fn_name=None,
        cov_kernel_name=None,
    ):
        """
        mean_fn: callable t -> tensor shape (d,)
        cov_kernel: callable (t, s) -> tensor shape (d, d)
        times: 1D iterable of ascending sampling times
        jitter: small diagonal regularizer for numerical stability
        """
        self.mean_fn = mean_fn
        self.cov_kernel = cov_kernel
        self.mean_fn_name, self.mean_fn_config = self._extract_registry_info(
            mean_fn,
            mean_fn_name,
            "mean",
        )
        self.cov_kernel_name, self.cov_kernel_config = self._extract_registry_info(
            cov_kernel,
            cov_kernel_name,
            "kernel",
        )
        self.jitter = jitter

        super().__init__(
            name="GaussianProcessSampler",
            mean_fn=self.mean_fn_name,
            mean_fn_config=self.mean_fn_config,
            cov_kernel=self.cov_kernel_name,
            cov_kernel_config=self.cov_kernel_config,
            times=list(times),
            jitter=jitter,
            ldl_decomp=ldl_decomp,
        )
        self.update_times(times, ldl_decomp=ldl_decomp)
    
    @staticmethod
    def _extract_registry_info(fn, override_name, kind):
        name = override_name or getattr(fn, "_registry_name", None)
        config = getattr(fn, "_registry_config", None)
        if name is None or config is None:
            raise ValueError(f"{kind} function is not registered.")
        return name, dict(config)

    def update_times(self, times, ldl_decomp=True):
        times = torch.as_tensor(times, dtype=torch.float32)
        self.times = times

        mean_blocks = [self.mean_fn(t) for t in times]
        self.mean_vals = torch.stack(mean_blocks, dim=0)  # (T, d)
        T, d = self.mean_vals.shape
        self.T, self.d = T, d

        mean_vec = self.mean_vals.reshape(-1)
        cov = mean_vec.new_zeros(T * d, T * d)
        for i, ti in enumerate(times):
            for j, tj in enumerate(times):
                cov[i*d:(i+1)*d, j*d:(j+1)*d] = self.cov_kernel(ti, tj)
        cov = cov + self.jitter * torch.eye(T * d, dtype=cov.dtype, device=cov.device)

        self.cov = 0.5 * (cov + cov.T)  # ensure symmetry
        if ldl_decomp:
            self.L, self.D = block_ldl(self.cov, d)
        else:
            self.L, self.D = None, None
        self.mvn = torch.distributions.MultivariateNormal(mean_vec, covariance_matrix=self.cov)

    def sample(self, batch_size, flatten=True):
        samples = self.mvn.sample((batch_size,))
        if not flatten:
            samples = samples.view(batch_size, self.T, self.d)
        return samples

    def get_stats(self, flatten=False):
        """Return μ(t₁),…,μ(t_T) as a tensor of shape (T, d) if not flattened or (Td) if flattened."""
        mean = self.mean_vals if not flatten else self.mean_vals.reshape(-1)
        return mean, self.cov

    def conditional_distribution(self, past_values=None, past_steps=None, whitened_past=None):
        """Return a `ConditionalPredictor` for the remaining time steps."""
        if self.L is None or self.D is None:
            raise RuntimeError("Block-LDL factors not cached. Recreate with ldl_decomp=True.")
        return conditional_from_block_ldl(
            self.L,
            self.D,
            self.mean_vals,
            past_values=past_values,
            past_steps=past_steps,
            whitened_past=whitened_past,
            block_dim=self.d,
        )

    def conditional_predict(self, past_values=None, past_steps=None, whitened_past=None,flatten=False):
        """Return only the conditional mean/covariance of future steps."""
        dist = self.conditional_distribution(
            past_values=past_values,
            past_steps=past_steps,
            whitened_past=whitened_past,
        )
        return dist.mean(flatten=flatten), dist.covariance

    def conditional_sample(self, num_samples, past_values=None, past_steps=None, whitened_past=None, flatten=False):
        """Sample conditional trajectories independently of the statistics call."""
        return self.conditional_distribution(
            past_values=past_values,
            past_steps=past_steps,
            whitened_past=whitened_past,
        ).sample(num_samples, flatten=flatten,)

    def recursive_conditioner(self, num_trajectories=1):
        """Return a stateful predictor for efficient sequential conditioning."""
        if self.L is None or self.D is None:
            raise RuntimeError("Block-LDL factors not cached. Recreate with ldl_decomp=True.")
        return RecursivePredictor(
            self.L,
            self.D,
            self.mean_vals,
            block_dim=self.d,
            num_trajectories=num_trajectories,
        )
    
    def state_dict(self):
        cfg = {
            "mean_fn": self.mean_fn_name,
            "mean_fn_config": self.mean_fn_config,
            "cov_kernel": self.cov_kernel_name,
            "cov_kernel_config": self.cov_kernel_config,
            "times": self.times.cpu().tolist(),
            "jitter": self.jitter,
        }
        return {"name": self.name, "config": cfg}

    @classmethod
    def from_state_dict(cls, state):
        cfg = state["config"]
        mean_fn = build_mean_fn(cfg["mean_fn"], cfg.get("mean_fn_config"))
        cov_kernel = build_kernel(cfg["cov_kernel"], cfg.get("cov_kernel_config"))
        return cls(
            mean_fn=mean_fn,
            cov_kernel=cov_kernel,
            times=cfg["times"],
            jitter=cfg.get("jitter", 1e-6),
            mean_fn_name=cfg["mean_fn"],
            cov_kernel_name=cfg["cov_kernel"],
            ldl_decomp=True,
        )



class FractionalNoiseSampler(Sampler):
    """
    Utility to generate fractional Brownian motion (fBm) paths and
    fractional Gaussian noise (increments) for multiple trajectories.
    """

    def __init__(self, hurst, t0=0.0, t1=1.0, steps=1000, device="cpu", mode="fbm"):
        super().__init__(
            name="FractionalNoiseSampler",
            hurst=float(hurst),
            t0=float(t0),
            t1=float(t1),
            steps=int(steps),
            device=str(device),
            mode=mode,
        )
        self.hurst = float(hurst)
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.steps = int(steps)
        self.device = device
        self.mode = mode.lower()
        self.times = torch.linspace(self.t0, self.t1, self.steps + 1, device=device)

    def _sample_fbm(self, batch_size):
        paths = []
        for _ in range(batch_size):
            fbm_path = FBM(n=self.steps, hurst=self.hurst).fbm()
            paths.append(torch.tensor(fbm_path, device=self.device))
        data = torch.stack(paths, dim=0)
        return self.times.clone(), data

    def _sample_fgn(self, batch_size):
        fgn_list = []
        for _ in range(batch_size):
            fgn = FBM(n=self.steps, hurst=self.hurst).fgn()
            fgn_list.append(torch.tensor(fgn, device=self.device))
        data = torch.stack(fgn_list, dim=0)
        return data  # shape (batch, steps)

    def sample_fbm(self, batch_size):
        """Return fBm paths of shape (batch_size, steps+1)."""
        return self._sample_fbm(batch_size)

    def sample_fgn(self, batch_size):
        """Return fractional Gaussian noise (increments) of shape (batch, steps)."""
        return self._sample_fgn(batch_size)
    
    def sample(self, batch_size):
        """Return either (times, fBm paths) or just fGn increments depending on mode."""
        if self.mode == "fbm":
            return self._sample_fbm(batch_size)
        elif self.mode == "fgn":
            return self._sample_fgn(batch_size)
        raise ValueError(f"Unknown mode '{self.mode}'. Choose 'fbm' or 'fgn'.")
    
    def state_dict(self):
        cfg = self.config.copy()
        cfg["mode"] = self.mode
        return {"name": self.name, "config": cfg}

    @classmethod
    def from_state_dict(cls, state):
        cfg = state["config"]
        return cls(
            hurst=cfg["hurst"],
            t0=cfg["t0"],
            t1=cfg["t1"],
            steps=cfg["steps"],
            device=cfg.get("device", "cpu"),
            mode=cfg.get("mode", "fbm"),
        )
    


##########################
# Block LDLᵀ factorization
##########################
def block_ldl(cov, block_dim):
    """
    Block LDLᵀ factorization with unit diagonal blocks in L.
    cov: (T*block_dim, T*block_dim) SPD matrix
    returns: (L, D), both (T*block_dim, T*block_dim)
    """
    total_dim = cov.shape[0]
    assert total_dim % block_dim == 0, "Cov size must be divisible by block_dim."
    T = total_dim // block_dim

    L = cov.new_zeros(total_dim, total_dim)
    D = cov.new_zeros(total_dim, total_dim)
    eye_block = torch.eye(block_dim, dtype=cov.dtype, device=cov.device)

    for k in range(T):
        k_slice = slice(k * block_dim, (k + 1) * block_dim)
        # Schur complement for current block
        Sk = cov[k_slice, k_slice].clone()
        for j in range(k):
            j_slice = slice(j * block_dim, (j + 1) * block_dim)
            Ljk = L[k_slice, j_slice]
            Dj = D[j_slice, j_slice]
            Sk -= Ljk @ Dj @ Ljk.T

        D[k_slice, k_slice] = Sk
        L[k_slice, k_slice] = eye_block

        D_inv = torch.linalg.inv(Sk)  # block inverse (d×d)
        for i in range(k + 1, T):
            i_slice = slice(i * block_dim, (i + 1) * block_dim)
            Sik = cov[i_slice, k_slice].clone()
            for j in range(k):
                j_slice = slice(j * block_dim, (j + 1) * block_dim)
                Lij = L[i_slice, j_slice]
                Ljk = L[k_slice, j_slice]
                Dj = D[j_slice, j_slice]
                Sik -= Lij @ Dj @ Ljk.T
            L[i_slice, k_slice] = Sik @ D_inv

    return L, D


def conditional_from_block_ldl(
    L,
    D,
    mean,
    past_values=None,
    past_steps=None,
    whitened_past=None,
    block_dim=None,
):
    """Return a conditional GP prediction object derived from block-LDL factors.

    Supports conditioning on multiple trajectories in parallel by passing
    ``past_values`` or ``whitened_past`` with a batch dimension.

    Parameters
    ----------
    past_values : Tensor, optional
        Observed past blocks. Accepted shapes:
            - (t, d)
            - (batch, t, d)
            - flattened equivalents with length ``t * d`` (batch dimension optional).
    past_steps : int, optional
        Number of conditioned steps when omitting ``past_values``.
    whitened_past : Tensor, optional
        Whitened residuals η solving ``L_pp η = y_p - μ_p``. May include a batch
        dimension to describe multiple trajectories.

    Returns
    -------
    ConditionalPredictor
        Object exposing ``mean()``, ``covariance``, and ``sample()`` for the
        future portion of the process.
    """

    mean = torch.as_tensor(mean)

    if mean.dim() == 2:
        T, inferred_dim = mean.shape
        block_dim = inferred_dim if block_dim is None else block_dim
        if block_dim != inferred_dim:
            raise ValueError("Provided block_dim does not match mean dimensionality.")
        mean_vals = mean
    elif mean.dim() == 1:
        if block_dim is None:
            raise ValueError("block_dim must be specified when mean is flattened.")
        total_dim = mean.shape[0]
        if total_dim % block_dim != 0:
            raise ValueError("Mean vector length must be divisible by block_dim.")
        T = total_dim // block_dim
        mean_vals = mean.view(T, block_dim)
    else:
        raise ValueError("Mean tensor must be 1D or 2D.")

    mean_vals = mean_vals.to(dtype=L.dtype, device=L.device)
    T = mean_vals.shape[0]

    dtype = mean_vals.dtype
    device = mean_vals.device

    batch_size = None

    def _ensure_batch(values, name):
        nonlocal batch_size
        if values is None:
            return None
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
        if tensor.numel() == 0:
            return None
        if tensor.dim() == 1:
            if tensor.shape[0] % block_dim != 0:
                raise ValueError(f"{name} length must be divisible by block_dim.")
            tensor = tensor.view(1, -1, block_dim)
        elif tensor.dim() == 2:
            if tensor.shape[1] == block_dim:
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[0] == block_dim and tensor.shape[1] % block_dim == 0:
                tensor = tensor.view(1, -1, block_dim)
            else:
                raise ValueError(f"Ambiguous shape for {name}; expected (..., block_dim).")
        elif tensor.dim() == 3:
            if tensor.shape[2] != block_dim:
                raise ValueError(f"Last dimension of {name} must equal block_dim.")
        else:
            raise ValueError(f"{name} must be a 1D, 2D, or 3D tensor.")

        current_batch = tensor.shape[0]
        if batch_size is None:
            batch_size = current_batch
        elif current_batch not in (1, batch_size):
            raise ValueError(f"Batch size mismatch for {name}.")
        if batch_size > 1 and current_batch == 1:
            tensor = tensor.expand(batch_size, -1, -1)
        return tensor

    past_tensor = _ensure_batch(past_values, "past_values")

    if whitened_past is not None:
        whitened = torch.as_tensor(whitened_past, dtype=dtype, device=device)
        if whitened.dim() == 1:
            whitened = whitened.unsqueeze(0)
        elif whitened.dim() > 2:
            raise ValueError("whitened_past must be 1D or 2D (batch, past_dim).")
        if batch_size is None:
            batch_size = whitened.shape[0]
        elif whitened.shape[0] not in (1, batch_size):
            raise ValueError("Batch size mismatch between whitened_past and past_values.")
        if whitened.shape[0] == 1 and batch_size > 1:
            whitened = whitened.expand(batch_size, -1)
        whitened = whitened.contiguous()
    else:
        whitened = None

    if batch_size is None:
        batch_size = 1

    empty_past = mean_vals.new_zeros(batch_size, 0, block_dim)
    if past_tensor is None:
        past_tensor = empty_past

    inferred_steps = past_tensor.shape[1]
    if whitened is not None and whitened.shape[1] % block_dim != 0:
        raise ValueError("whitened_past length must be a multiple of block_dim.")
    whitened_steps = None if whitened is None else whitened.shape[1] // block_dim

    if past_steps is None:
        t = inferred_steps if inferred_steps > 0 else (whitened_steps or 0)
    else:
        t = int(past_steps)
    if t < 0:
        raise ValueError("past_steps must be non-negative.")
    if t > T:
        raise ValueError("Cannot condition on more observations than process length.")
    if inferred_steps not in (0, t):
        raise ValueError("past_values must contain exactly t steps or be empty.")
    if whitened_steps is not None and whitened_steps != t:
        raise ValueError("whitened_past length inconsistent with past_steps.")

    total_dim = T * block_dim
    past_dim = t * block_dim
    future_blocks = T - t
    future_dim = future_blocks * block_dim

    future_slice = slice(past_dim, total_dim)
    mean_future = mean_vals[t:].reshape(-1)  # (future_dim,)

    L_ff = L[future_slice, future_slice]
    D_ff = D[future_slice, future_slice]
    cov_cond = L_ff @ (D_ff @ L_ff.T)
    cov_cond = 0.5 * (cov_cond + cov_cond.T)

    mean_future = mean_future.unsqueeze(0).expand(batch_size, -1)

    if t == 0:
        cond_mean_flat = mean_future
    else:
        past_slice = slice(0, past_dim)
        L_fp = L[future_slice, past_slice]
        if whitened is None:
            if inferred_steps == 0:
                raise ValueError("past_values required to compute whitened residuals.")
            L_pp = L[past_slice, past_slice]
            residual = past_tensor[:, :t, :] - mean_vals[:t].unsqueeze(0)
            rhs = residual.reshape(batch_size, past_dim).T
            eta = torch.linalg.solve_triangular(
                L_pp,
                rhs,
                upper=False,
                unitriangular=True,
            ).T
        else:
            eta = whitened[:, :past_dim]
        cond_mean_flat = mean_future + (L_fp @ eta.T).T

    return ConditionalPredictor(
        cond_mean_flat=cond_mean_flat,
        covariance=cov_cond,
        L_future=L_ff,
        D=D,
        block_dim=block_dim,
        start_step=t,
        future_blocks=future_blocks,
        batch_size=batch_size,
    )


class ConditionalPredictor:
    """Encapsulates conditional GP statistics and provides sampling utilities."""

    def __init__(
        self,
        cond_mean_flat,
        covariance,
        L_future,
        D,
        block_dim,
        start_step,
        future_blocks,
        batch_size,
    ):
        self._mean_flat = cond_mean_flat.contiguous() if cond_mean_flat is not None else cond_mean_flat
        self._covariance = covariance
        self._L_future = L_future
        self._D = D
        self.block_dim = int(block_dim)
        self.start_step = int(start_step)
        self.future_blocks = int(future_blocks)
        self.future_dim = self.future_blocks * self.block_dim
        self.batch_size = int(batch_size)

    def mean(self, flatten=False):
        if self._mean_flat is None:
            return None
        if flatten:
            output = self._mean_flat
        else:
            output = self._mean_flat.view(self.batch_size, self.future_blocks, self.block_dim)
        if self.batch_size == 1:
            output = output.squeeze(0)
        return output

    @property
    def covariance(self):
        return self._covariance

    def sample(self, num_samples, flatten=False):
        num_samples = int(num_samples)
        dtype = self._mean_flat.dtype
        device = self._mean_flat.device
        white = torch.randn(
            num_samples,
            self.batch_size,
            self.future_blocks,
            self.block_dim,
            dtype=dtype,
            device=device,
        )
        block_white = torch.empty_like(white)
        for idx in range(self.future_blocks):
            block_idx = self.start_step + idx
            full_slice = slice(block_idx * self.block_dim, (block_idx + 1) * self.block_dim)
            D_block = self._D[full_slice, full_slice]
            D_block = 0.5 * (D_block + D_block.T)
            chol = torch.linalg.cholesky(D_block)
            block_white[:, :, idx, :] = white[:, :, idx, :] @ chol.T
        eps = block_white.reshape(num_samples * self.batch_size, self.future_dim)
        eps = eps @ self._L_future.T
        eps = eps.view(num_samples, self.batch_size, self.future_dim)
        samples = eps + self._mean_flat.unsqueeze(0)
        if not flatten:
            samples = samples.view(num_samples, self.batch_size, self.future_blocks, self.block_dim)
        if self.batch_size == 1:
            samples = samples.squeeze(1)
        return samples


class RecursivePredictor:
    """Stateful helper for batch recursive GP conditioning using block-LDL factors."""

    def __init__(self, L, D, mean, block_dim=None, num_trajectories=1):
        mean = torch.as_tensor(mean)
        if mean.dim() == 1:
            if block_dim is None:
                raise ValueError("block_dim is required when mean is flattened.")
            if mean.shape[0] % block_dim != 0:
                raise ValueError("Mean length must be divisible by block_dim.")
            self.mean_vals = mean.view(-1, block_dim).to(dtype=L.dtype, device=L.device)
        elif mean.dim() == 2:
            d = mean.shape[1]
            if block_dim is not None and block_dim != d:
                raise ValueError("block_dim does not match mean dimensionality.")
            block_dim = d
            self.mean_vals = mean.to(dtype=L.dtype, device=L.device)
        else:
            raise ValueError("mean must be rank-1 or rank-2 tensor.")

        self.L = L
        self.D = D
        self.block_dim = block_dim
        self.total_steps = self.mean_vals.shape[0]
        self.num_trajectories = int(num_trajectories)
        if self.num_trajectories <= 0:
            raise ValueError("num_trajectories must be positive.")
        self.reset(num_trajectories=self.num_trajectories)

    def reset(self, num_trajectories=None):
        if num_trajectories is not None:
            self.num_trajectories = int(num_trajectories)
            if self.num_trajectories <= 0:
                raise ValueError("num_trajectories must be positive.")
        self.t = 0
        self._eta = self.mean_vals.new_zeros(self.num_trajectories, 0)

    @property
    def conditioned_steps(self):
        return self.t

    @property
    def whitened_past(self):
        return self._eta.clone()

    def observe(self, values):
        values = torch.as_tensor(values, dtype=self.mean_vals.dtype, device=self.mean_vals.device)
        if values.dim() == 1:
            if self.num_trajectories != 1 or values.shape[0] != self.block_dim:
                raise ValueError("Observation dimension mismatch.")
            values = values.view(1, 1, self.block_dim)
        elif values.dim() == 2:
            if values.shape[1] != self.block_dim:
                raise ValueError("Values must have last dimension block_dim.")
            if self.num_trajectories == 1:
                values = values.unsqueeze(0)
            elif values.shape[0] == self.num_trajectories:
                values = values.unsqueeze(1)
            else:
                raise ValueError("Batch dimension mismatch in observe().")
        elif values.dim() == 3:
            if values.shape[0] != self.num_trajectories or values.shape[2] != self.block_dim:
                raise ValueError("Values must have shape (batch, steps, block_dim).")
        else:
            raise ValueError("Values must be 1D, 2D, or 3D tensor.")

        steps = values.shape[1]
        for step in range(steps):
            if self.t >= self.total_steps:
                raise ValueError("All time steps already conditioned.")
            obs = values[:, step, :]
            residual = obs - self.mean_vals[self.t]
            row_slice = slice(self.t * self.block_dim, (self.t + 1) * self.block_dim)
            if self.t == 0:
                eta_new = residual
            else:
                coupling = self._eta @ self.L[row_slice, : self.t * self.block_dim].T
                eta_new = residual - coupling
            eta_new = eta_new.reshape(self.num_trajectories, self.block_dim)
            self._eta = torch.cat([self._eta, eta_new], dim=1)
            self.t += 1
        return self

    def distribution(self):
        return conditional_from_block_ldl(
            self.L,
            self.D,
            self.mean_vals,
            past_values=None,
            past_steps=self.t,
            whitened_past=self._eta,
            block_dim=self.block_dim,
        )

    def predict(self, flatten=False):
        dist = self.distribution()
        return dist.mean(flatten=flatten), dist.covariance

    def sample(self, num_samples, flatten=False):
        return self.distribution().sample(num_samples, flatten=flatten)



__all__ = [
    "GaussianVectorSampler",
    "GPSampler",
    "FractionalNoiseSampler",
    "block_ldl",
    "conditional_from_block_ldl",
    "ConditionalPredictor",
    "RecursivePredictor",
]
