import math
import numpy as np
import torch
import functools


__all__ = [
    "MEAN_FN_REGISTRY",
    "register_mean_fn",
    "build_mean_fn",
    "spiral_mean",
    "zero_mean",
    "KERNEL_REGISTRY",
    "register_kernel",
    "build_kernel",
    "rbf_kernel",
    "laplace_kernel",
    "exponential_kernel",
    "matern_kernel",
    "rational_quadratic_kernel",
    "periodic_kernel",
    "locally_periodic_kernel",
    "polynomial_kernel",
    "brownian_kernel",
    "fractional_brownian_kernel",
]


def _tag_registry_callable(func, name, config):
    func._registry_name = name
    func._registry_config = dict(config)


##########################
# Mean functions registry
##########################
MEAN_FN_REGISTRY = {}
def register_mean_fn(name):
    def decorator(fn):
        def factory(**config):
            def wrapped(t):
                return fn(t, **config)
            _tag_registry_callable(wrapped, name, config)
            return wrapped
        MEAN_FN_REGISTRY[name] = factory
        _tag_registry_callable(fn, name, {})
        return fn
    return decorator


@register_mean_fn("spiral")
def spiral_mean(t, freq=1.0):
    amp = 1.0 + 0.5 * torch.sin(6 * 2 * torch.pi * t)
    angle = freq * torch.pi * t
    return amp * torch.stack([torch.cos(angle), torch.sin(angle)])

@register_mean_fn("zero")
def zero_mean(t, dim=2):
    return torch.zeros(dim, device=t.device, dtype=t.dtype)


##########################
# Kernel functions registry
##########################
KERNEL_REGISTRY = {}
def register_kernel(name):
    def decorator(fn):
        def factory(**config):
            def wrapped(t, s, *args, **kwargs):
                params = {**config, **kwargs}
                return fn(t, s, *args, **params)
            _tag_registry_callable(wrapped, name, config)
            return wrapped
        KERNEL_REGISTRY[name] = factory
        _tag_registry_callable(fn, name, {})
        return fn
    return decorator


def build_mean_fn(name, config=None):
    factory = MEAN_FN_REGISTRY[name]
    config = config or {}
    return factory(**config)


def build_kernel(name, config=None):
    factory = KERNEL_REGISTRY[name]
    config = config or {}
    return factory(**config)


@register_kernel("rbf")
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

@register_kernel("laplace")
def laplace_kernel(t, s, sigma=1.0, tau=0.1, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-dt / tau)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))

@register_kernel("exponential")
def exponential_kernel(t, s, sigma=1.0, tau=0.1, beta=0.5, Cov=None):
    """
    Generalized exponential kernel with power beta (0 < beta ≤ 2).
    """
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-(dt / tau) ** beta)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))

@register_kernel("matern")
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

@register_kernel("rational_quadratic")
def rational_quadratic_kernel(t, s, sigma=1.0, tau=0.1, alpha=1.0, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * (1 + (dt ** 2) / (2 * alpha * tau ** 2)) ** (-alpha)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))

@register_kernel("periodic")
def periodic_kernel(t, s, sigma=1.0, tau=0.1, period=1.0, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-2 * torch.sin(torch.pi * dt / period) ** 2 / tau ** 2)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))

@register_kernel("locally_periodic")
def locally_periodic_kernel(t, s, sigma=1.0, tau=0.1, period=1.0, Cov=None):
    dt = torch.abs(t - s)
    k = sigma ** 2 * torch.exp(-dt ** 2 / (2 * tau ** 2)) \
        * torch.exp(-2 * torch.sin(torch.pi * dt / period) ** 2 / tau ** 2)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))

@register_kernel("polynomial")
def polynomial_kernel(t, s, c=1.0, degree=2, sigma=1.0, Cov=None):
    k = sigma ** 2 * (t * s + c) ** degree
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


@register_kernel("brownian")
def brownian_kernel(t, s, sigma=1.0, Cov=None):
    """
    Covariance of standard Brownian motion: k(t, s) = sigma^2 * min(t, s).
    """
    t_abs = torch.abs(t)
    s_abs = torch.abs(s)
    k = sigma ** 2 * torch.minimum(t_abs, s_abs)
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))


@register_kernel("fractional_brownian")
def fractional_brownian_kernel(t, s, hurst=0.5, sigma=1.0, Cov=None):
    """
    Fractional Brownian motion covariance:
    k(t, s) = 0.5 * sigma^2 * (|t|^{2H} + |s|^{2H} - |t - s|^{2H})
    """
    t_abs = torch.abs(t)
    s_abs = torch.abs(s)
    dt = torch.abs(t - s)
    k = 0.5 * (sigma ** 2) * (t_abs ** (2 * hurst) + s_abs ** (2 * hurst) - dt ** (2 * hurst))
    return k * (Cov if Cov is not None else torch.eye(1, device=t.device))
