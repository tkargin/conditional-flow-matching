import math
import numpy as np
import torch

__all__ = [
    "DRIFT_REGISTRY",
    "DIFFUSION_REGISTRY",
    "SDE_REGISTRY",
    "register_drift",
    "register_diffusion",
    "register_sde",
    "build_sde",
    "ou_system",
    "lorenz_system",
    "rossler_system",
    "vanderpol_system",
    "duffing_system",
    "chua_system",
]

DRIFT_REGISTRY = {}
DIFFUSION_REGISTRY = {}
SDE_REGISTRY = {}

def register_drift(name):
    def decorator(fn):
        DRIFT_REGISTRY[name] = fn
        return fn
    return decorator

def register_diffusion(name):
    def decorator(fn):
        DIFFUSION_REGISTRY[name] = fn
        return fn
    return decorator

def register_sde(name):
    def decorator(system_fn):
        SDE_REGISTRY[name] = system_fn
        return system_fn
    return decorator

def build_sde(name, config=None):
    config = config or {}
    return SDE_REGISTRY[name](**config)

def _tag_sde_callable(func, name, config, role):
    func._sde_name = name
    func._sde_role = role
    func._sde_config = dict(config)

##############################
# 1. Ornstein–Uhlenbeck (OU) #
##############################
@register_sde("ou")
def ou_system(theta=0.7, mu=None, sigma=0.3, dim=2):
    mu_vec = torch.zeros(dim) if mu is None else torch.as_tensor(mu, dtype=torch.float32)
    def drift_fn(t, x):
        return theta * (mu_vec.to(x.device) - x)
    def diffusion_fn(t, x):
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return sigma * eye.unsqueeze(0).expand(x.size(0), dim, dim)
    params = {
        "theta": theta,
        "mu": mu_vec.tolist() if isinstance(mu_vec, torch.Tensor) else mu,
        "sigma": sigma,
        "dim": dim,
    }
    _tag_sde_callable(drift_fn, "ou", params, "drift")
    _tag_sde_callable(diffusion_fn, "ou", params, "diffusion")
    return drift_fn, diffusion_fn

##################
# 2. Lorenz 63   #
##################
@register_sde("lorenz")
def lorenz_system(sigma=10.0, rho=28.0, beta=8.0 / 3.0, noise_scale=0.5):
    def drift_fn(t, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        return torch.stack([dx1, dx2, dx3], dim=1)
    def diffusion_fn(t, x):
        scale = noise_scale + 0.05 * torch.norm(x, dim=1, keepdim=True)
        eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
        return eye * scale.unsqueeze(-1)
    params = {
        "sigma": sigma,
        "rho": rho,
        "beta": beta,
        "noise_scale": noise_scale,
    }
    _tag_sde_callable(drift_fn, "lorenz", params, "drift")
    _tag_sde_callable(diffusion_fn, "lorenz", params, "diffusion")
    return drift_fn, diffusion_fn

####################
# 3. Rössler 1976  #
####################
@register_sde("rossler")
def rossler_system(a=0.2, b=0.2, c=5.7, noise_scale=(0.3, 0.3, 0.3)):
    noise = torch.as_tensor(noise_scale, dtype=torch.float32)
    def drift_fn(t, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        dx1 = -x2 - x3
        dx2 = x1 + a * x2
        dx3 = b + x3 * (x1 - c)
        return torch.stack([dx1, dx2, dx3], dim=1)
    def diffusion_fn(t, x):
        return torch.diag_embed(noise.to(x.device)).unsqueeze(0).expand(x.size(0), 3, 3)
    params = {
        "a": a,
        "b": b,
        "c": c,
        "noise_scale": noise.tolist(),
    }
    _tag_sde_callable(drift_fn, "rossler", params, "drift")
    _tag_sde_callable(diffusion_fn, "rossler", params, "diffusion")
    return drift_fn, diffusion_fn

######################
# 4. van der Pol     #
######################
@register_sde("vanderpol")
def vanderpol_system(mu=1.0, omega=2 * torch.pi, noise_scale=(0.05, 0.05)):
    noise = torch.as_tensor(noise_scale, dtype=torch.float32)
    def drift_fn(t, x):
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = x2
        dx2 = mu * (1 - x1 ** 2) * x2 - omega ** 2 * x1
        return torch.stack([dx1, dx2], dim=1)
    def diffusion_fn(t, x):
        return torch.diag_embed(noise.to(x.device)).unsqueeze(0).expand(x.size(0), 2, 2)
    params = {
        "mu": mu,
        "omega": float(omega),
        "noise_scale": noise.tolist(),
    }
    _tag_sde_callable(drift_fn, "vanderpol", params, "drift")
    _tag_sde_callable(diffusion_fn, "vanderpol", params, "diffusion")
    return drift_fn, diffusion_fn

##########################
# 5. Duffing Oscillator  #
##########################
@register_sde("duffing")
def duffing_system(delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2, noise=0.1):
    def drift_fn(t, x):
        pos, vel = x[:, 0], x[:, 1]
        dpos = vel
        dvel = gamma * torch.cos(omega * t) - delta * vel - alpha * pos - beta * pos ** 3
        return torch.stack([dpos, dvel], dim=1)
    def diffusion_fn(t, x):
        return torch.diag_embed(torch.tensor([0.0, noise], device=x.device, dtype=x.dtype)).unsqueeze(0).expand(x.size(0), 2, 2)
    params = {
        "delta": delta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "omega": omega,
        "noise": noise,
    }
    _tag_sde_callable(drift_fn, "duffing", params, "drift")
    _tag_sde_callable(diffusion_fn, "duffing", params, "diffusion")
    return drift_fn, diffusion_fn

#################################
# 6. Chua's Circuit (chaotic)   #
#################################
@register_sde("chua")
def chua_system(alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714, noise_scale=0.2):
    def nonlinearity(x):
        return m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1) - torch.abs(x - 1))
    def drift_fn(t, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        dx1 = alpha * (x2 - x1 - nonlinearity(x1))
        dx2 = x1 - x2 + x3
        dx3 = -beta * x2
        return torch.stack([dx1, dx2, dx3], dim=1)
    def diffusion_fn(t, x):
        return torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0) * noise_scale
    params = {
        "alpha": alpha,
        "beta": beta,
        "m0": m0,
        "m1": m1,
        "noise_scale": noise_scale,
    }
    _tag_sde_callable(drift_fn, "chua", params, "drift")
    _tag_sde_callable(diffusion_fn, "chua", params, "diffusion")
    return drift_fn, diffusion_fn
