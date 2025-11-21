import math
import numpy as np
import torch
import torch.nn as nn
import torchsde

from .base_sampler import Sampler
from .sde_registry import build_sde



##########################
# TorchSDE-based Sampler Class
##########################
class SDESampler(Sampler):
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
        drift_name=None,
        diffusion_name=None,
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
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.drift_name, self.sde_config = self._extract_registry_info(
            drift_fn,
            drift_name,
            "drift",
        )
        self.diffusion_name, _ = self._extract_registry_info(
            diffusion_fn,
            diffusion_name,
            "diffusion",
        )
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.steps = int(steps)
        self.times = torch.linspace(self.t0, self.t1, self.steps + 1)
        self.init_sampler = init_sampler
        self.method = method
        self.adaptive = adaptive
        self.solver_kwargs = solver_kwargs

        super().__init__(
            name="SDESampler",
            drift=self.drift_name,
            diffusion=self.diffusion_name,
            sde_config=self.sde_config,
            t0=self.t0,
            t1=self.t1,
            steps=self.steps,
            method=self.method,
            adaptive=self.adaptive,
            solver_kwargs=self.solver_kwargs,
        )

        class _SDE(nn.Module):
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

        class _Deterministic(nn.Module):
            noise_type = "general"
            sde_type = "ito"
            def __init__(self, drift):
                super().__init__()
                self._drift = drift
            def f(self, t, x):
                return self._drift(t, x)
            def g(self, t, x):
                return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)

        self.sde = _SDE(self.drift_fn, self.diffusion_fn)
        self.det_sde = _Deterministic(self.drift_fn)

    @staticmethod
    def _extract_registry_info(fn, override_name, kind):
        name = override_name or getattr(fn, "_sde_name", None)
        config = getattr(fn, "_sde_config", None)
        if name is None or config is None:
            raise ValueError(f"{kind} function is not registered.")
        return name, dict(config)

    @torch.no_grad()
    def sample(self, batch_size, x0=None, device="cpu", flatten=False, init_mean=None):
        """
        Returns
        -------
        trajectories : (batch_size, steps+1, d) tensor (or flattened if requested)
        times : (steps+1,) tensor of evaluation times
        mean_traj : (steps+1, d) tensor approximating E[X_t]
        """
        if self.init_sampler is not None:
            x_init = self.init_sampler(batch_size).to(device)
        else:
            if x0 is None:
                raise ValueError("Provide x0 when init_sampler is None.")
            x_init = x0.expand(batch_size, -1).to(device)

        times = self.times.to(device)
        traj = torchsde.sdeint(self.sde, x_init, times, method=self.method, adaptive=self.adaptive, **self.solver_kwargs)
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
            return traj.view(batch_size, -1), times, mean_traj
        return traj, times, mean_traj
    
    def state_dict(self):
        cfg = self.config.copy()
        cfg["drift"] = self.drift_name
        cfg["diffusion"] = self.diffusion_name
        cfg["sde_config"] = self.sde_config
        return {"name": self.name, "config": cfg}

    @classmethod
    def from_state_dict(cls, state):
        cfg = state["config"]
        drift_fn, diffusion_fn = build_sde(cfg["drift"], cfg.get("sde_config"))
        solver_kwargs = cfg.get("solver_kwargs", {})
        return cls(
            drift_fn=drift_fn,
            diffusion_fn=diffusion_fn,
            t0=cfg["t0"],
            t1=cfg["t1"],
            steps=cfg["steps"],
            method=cfg.get("method", "euler"),
            adaptive=cfg.get("adaptive", False),
            drift_name=cfg["drift"],
            diffusion_name=cfg["diffusion"],
            **solver_kwargs,
        )


__all__ = ["SDESampler"]
