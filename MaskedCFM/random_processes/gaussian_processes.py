import math
import numpy as np
import torch
import torch.nn as nn
from fbm import FBM


##########################
# Kernel functions
##########################
def rbf_kernel(t, s, sigma = 1.0, tau = 0.1, Cov = None):
    """
    Radial Basis Function (RBF) kernel (squared exponential kernel).
    k(t, s) = sigma^2 * exp(-|t - s|^2 / (2 * tau^2)) * Cov
    where Cov is a positive semi-definite matrix.
    Parameters
    ----------
    t : Tensor
        First input (scalar time).
    s : Tensor
        Second input (scalar time).
    sigma : float
        Signal variance.
    tau : float
        Lengthscale.
    Cov : Tensor
        Positive semi-definite matrix for output covariance.
    """
    dt = torch.abs(t - s)
    k_ts = (sigma**2) * torch.exp(-dt**2 / (2 * tau**2))  # scalar SE kernel
    return k_ts *  (Cov if Cov is not None else torch.eye(1, device=t.device))


def laplace_kernel(t, s, sigma=1.0, tau=0.1, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-dt / tau)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


def exponential_quadratic_kernel(t, s, sigma=1.0, tau=0.1, beta=0.5, Cov=None):
    """
    Generalized exponential kernel with power beta (0 < beta ≤ 2).
    """
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-(dt / tau) ** beta)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


def matern_kernel(t, s, nu=1.5, sigma=1.0, tau=0.1, Cov=None):
    """
    Matérn kernel with smoothness nu ∈ {0.5, 1.5, 2.5,...}.
    """
    dt = torch.abs(t - s)
    if nu == 0.5:
        k = sigma ** 2 * torch.exp(-dt / tau)
    elif nu == 1.5:
        factor = torch.sqrt(torch.tensor(3.0)) * dt / tau
        k = sigma ** 2 * (1 + factor) * torch.exp(-factor)
    elif nu == 2.5:
        factor = torch.sqrt(torch.tensor(5.0)) * dt / tau
        k = sigma ** 2 * (1 + factor + factor**2 / 3) * torch.exp(-factor)
    else:
        # general form via modified Bessel function (may be slower)
        from torch.special import kv
        r = dt + 1e-12
        coef = (2 ** (1 - nu)) / torch.special.gamma(torch.tensor(nu))
        factor = torch.sqrt(torch.tensor(2 * nu)) * r / tau
        k = sigma ** 2 * coef * (factor ** nu) * kv(nu, factor)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


def rational_quadratic_kernel(t, s, sigma=1.0, tau=0.1, alpha=1.0, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * (1 + (dt ** 2) / (2 * alpha * tau ** 2)) ** (-alpha)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


def periodic_kernel(t, s, sigma=1.0, tau=0.1, period=1.0, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-2 * torch.sin(torch.pi * dt / period) ** 2 / tau ** 2)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


def locally_periodic_kernel(t, s, sigma=1.0, tau=0.1, period=1.0, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-dt ** 2 / (2 * tau ** 2)) \
        * torch.exp(-2 * torch.sin(torch.pi * dt / period) ** 2 / tau ** 2)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


def polynomial_kernel(t, s, c=1.0, degree=2, sigma=1.0, Cov=None):
    k = sigma ** 2 * (t * s + c) ** degree
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))





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



##########################
# GP Samplers Class
##########################
class GaussianProcessSampler:
    """
    Efficient GP sampler that caches mean, covariance, and block-LDL factors
    for a given set of sampling times. You can update the time grid later
    without rebuilding the entire class.
    """

    def __init__(self, mean_fn, cov_kernel, times, jitter=1e-6, ldl_decomp=True):
        """
        mean_fn: callable t -> tensor shape (d,)
        cov_kernel: callable (t, s) -> tensor shape (d, d)
        times: 1D iterable of ascending sampling times
        jitter: small diagonal regularizer for numerical stability
        """
        self.mean_fn = mean_fn
        self.cov_kernel = cov_kernel
        self.jitter = jitter
        self.update_times(times, ldl_decomp=ldl_decomp)

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
        #self.L, self.D = block_ldl(cov, d) if ldl_decomp else None, None
        self.mvn = torch.distributions.MultivariateNormal(mean_vec, covariance_matrix=cov)

    def sample(self, num_samples, flatten=False):
        samples = self.mvn.sample((num_samples,))  # (N, T*d)
        if not flatten:
            samples = samples.view(num_samples, self.T, self.d)
        return samples

    def get_stats(self, flatten=False):
        """Return μ(t₁),…,μ(t_T) as a tensor of shape (T, d) if not flattened or (Td) if flattened."""
        mean = self.mean_vals if not flatten else self.mean_vals.reshape(-1)
        return mean, self.cov

    #def get_ldl(self):
    #    """Return block-LDL factors (L, D) with d×d identity blocks on L’s diagonal."""
    #    return self.L, self.D



class FractionalNoiseGenerator:
    """
    Utility to generate fractional Brownian motion (fBm) paths and
    fractional Gaussian noise (increments) for multiple trajectories.
    """

    def __init__(self, hurst, t0=0.0, t1=1.0, steps=1000, device="cpu"):
        self.hurst = hurst
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.steps = steps
        self.device = device
        self.times = torch.linspace(self.t0, self.t1, steps + 1, device=device)

    def sample_fbm(self, batch_size):
        """Return fBm paths of shape (batch_size, steps+1)."""
        paths = []
        for _ in range(batch_size):
            fbm_path = FBM(n=self.steps, hurst=self.hurst).fbm()  # numpy
            paths.append(torch.tensor(fbm_path, device=self.device))
        paths = torch.stack(paths, dim=0)
        return self.times.clone(), paths

    def sample_fgn(self, batch_size):
        """Return fractional Gaussian noise (increments) of shape (batch, steps)."""
        fgn_list = []
        for _ in range(batch_size):
            fgn = FBM(n=self.steps, hurst=self.hurst).fgn()
            fgn_list.append(torch.tensor(fgn, device=self.device))
        return torch.stack(fgn_list, dim=0)  # (batch, steps)


