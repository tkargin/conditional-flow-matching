import math
import numpy as np
import torch
import torch.nn as nn
import torchsde


##############################
# 1. Ornstein–Uhlenbeck (OU) #
##############################
def make_ou_system(theta=0.7, mu=None, sigma=0.3, dim=2):
    """
    dX_t = theta*(mu - X_t) dt + sigma dW_t
    """
    mu = torch.zeros(dim) if mu is None else torch.as_tensor(mu, dtype=torch.float32)

    def drift_fn(t, x):
        return theta * (mu.to(x.device) - x)

    def diffusion_fn(t, x):
        eye = torch.eye(dim, device=x.device, dtype=x.dtype)
        return sigma * eye.unsqueeze(0).expand(x.size(0), dim, dim)

    return drift_fn, diffusion_fn


##################
# 2. Lorenz 63   #
##################
def make_lorenz_system(sigma=10.0, rho=28.0, beta=8.0 / 3.0, noise_scale=0.5):
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

    return drift_fn, diffusion_fn


####################
# 3. Rössler 1976  #
####################
def make_rossler_system(a=0.2, b=0.2, c=5.7, noise_scale=(0.3, 0.3, 0.3)):
    noise = torch.as_tensor(noise_scale, dtype=torch.float32)

    def drift_fn(t, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        dx1 = -x2 - x3
        dx2 = x1 + a * x2
        dx3 = b + x3 * (x1 - c)
        return torch.stack([dx1, dx2, dx3], dim=1)

    def diffusion_fn(t, x):
        return torch.diag_embed(noise.to(x.device)).unsqueeze(0).expand(x.size(0), 3, 3)

    return drift_fn, diffusion_fn


######################
# 4. van der Pol + noise #
######################
def make_vanderpol_system(mu=1.0, omega=2 * torch.pi, noise_scale=(0.05, 0.05)):
    noise = torch.as_tensor(noise_scale, dtype=torch.float32)

    def drift_fn(t, x):
        x1, x2 = x[:, 0], x[:, 1]
        dx1 = x2
        dx2 = mu * (1 - x1 ** 2) * x2 - omega ** 2 * x1
        return torch.stack([dx1, dx2], dim=1)

    def diffusion_fn(t, x):
        return torch.diag_embed(noise.to(x.device)).unsqueeze(0).expand(x.size(0), 2, 2)

    return drift_fn, diffusion_fn


##########################
# 5. Duffing Oscillator  #
##########################
def make_duffing_system(delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2, noise=0.1):
    """
    d^2x/dt^2 + delta*dx/dt + alpha*x + beta*x^3 = gamma*cos(omega*t) + noise
    """
    def drift_fn(t, x):
        pos, vel = x[:, 0], x[:, 1]
        dpos = vel
        dvel = gamma * torch.cos(omega * t) - delta * vel - alpha * pos - beta * pos ** 3
        return torch.stack([dpos, dvel], dim=1)

    def diffusion_fn(t, x):
        return torch.diag_embed(torch.tensor([0.0, noise], device=x.device, dtype=x.dtype)).unsqueeze(0).expand(x.size(0), 2, 2)

    return drift_fn, diffusion_fn


#################################
# 6. Chua's Circuit (chaotic)   #
#################################
def make_chua_system(alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714, noise_scale=0.2):
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

    return drift_fn, diffusion_fn




##########################
# SDE-based Sampler Class
##########################
class TorchSDEDiffusionSampler:
    """
    TorchSDE-based sampler that also provides an approximate mean trajectory
    given the mean of the initialization.
    """

    def __init__(
        self,
        drift_fn,
        diffusion_fn,
        t0,
        t1,
        steps,
        init_sampler=None,
        method="euler",
        adaptive=False,
        **solver_kwargs,
    ):
        """
        drift_fn: callable b(t, x) -> (batch, d)
        diffusion_fn: callable sigma(t, x) -> (batch, d, m)
        t0, t1: time interval bounds
        steps: number of evaluation points (fixed grid)
        init_sampler: callable returning (batch, d) initial states (optional)
        method: torchsde solver name
        adaptive: use adaptive solver if supported
        solver_kwargs: forwarded to torchsde.sdeint
        """
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.steps = steps
        self.times = torch.linspace(self.t0, self.t1, steps + 1)
        self.init_sampler = init_sampler
        self.method = method
        self.adaptive = adaptive
        self.solver_kwargs = solver_kwargs
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn

        class _SDE(torch.nn.Module):
            noise_type = "general"
            sde_type = "ito"

            def __init__(self, drift, diffusion):
                super().__init__()
                self._drift = drift
                self._diffusion = diffusion

            def f(self, t, x):
                return self._drift(t, x)

            def g(self, t, x):
                return self._diffusion(t, x)

        class _Deterministic(torch.nn.Module):
            noise_type = "general"
            sde_type = "ito"

            def __init__(self, drift):
                super().__init__()
                self._drift = drift

            def f(self, t, x):
                return self._drift(t, x)

            def g(self, t, x):
                return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        self.sde = _SDE(drift_fn, diffusion_fn)
        self.det_sde = _Deterministic(drift_fn)

    @torch.no_grad()
    def sample(self, num_paths, x0=None, device="cpu", flatten=False, init_mean=None):
        """
        Returns
        -------
        trajectories : (num_paths, steps+1, d) tensor (or flattened if requested)
        times : (steps+1,) tensor of evaluation times
        mean_traj : (steps+1, d) tensor approximating E[X_t]
        """
        if self.init_sampler is not None:
            x_init = self.init_sampler(num_paths).to(device)
        else:
            if x0 is None:
                raise ValueError("Provide x0 when init_sampler is None.")
            x_init = x0.expand(num_paths, -1).to(device)

        times = self.times.to(device)
        traj = torchsde.sdeint(
            self.sde,
            x_init,
            times,
            method=self.method,
            adaptive=self.adaptive,
            **self.solver_kwargs,
        )
        traj = traj.transpose(0, 1).contiguous()

        if init_mean is None:
            init_mean = x_init.mean(dim=0, keepdim=True)
        else:
            init_mean = init_mean.to(device).unsqueeze(0)

        mean_traj = torchsde.sdeint(
            self.det_sde,
            init_mean,
            times,
            method=self.method,
            adaptive=self.adaptive,
            **self.solver_kwargs,
        ).squeeze(1)

        if flatten:
            return traj.view(num_paths, -1), times, mean_traj
        return traj, times, mean_traj


