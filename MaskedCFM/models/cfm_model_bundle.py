import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, Any


__all__ = ["CFMModelBundle"]


class CFMModelBundle:
    """
    High-level wrapper around a conditional flow-matching model.

    Parameters
    ----------
    model : nn.Module
        Vector field taking concatenated [x_t, t] inputs.
    cond_path_fn : Callable
        Function generating conditional paths, e.g. SampleConditionalNoisyStraightPath.
    cond_vec_field_fn : Callable
        Function computing target conditional vector field, e.g. ConditionalVelocityField.
    prior_sampler : Sampler, optional
    target_sampler : Sampler, optional
    joint_sampler : Sampler, optional
        If provided, must return (x0, x1) jointly. Overrides prior/target samplers.
    ode_factory : Callable, optional
        Function taking a model and returning a NeuralODE-like object with .trajectory().
    sigma : float
        Noise hyperparameter passed to cond_path_fn.
    device : str or torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        cond_path_fn: Callable,
        cond_vec_field_fn: Callable,
        prior_sampler=None,
        target_sampler=None,
        joint_sampler=None,
        ode_factory: Optional[Callable[[nn.Module], Any]] = None,
        device="cpu",
    ):
        if joint_sampler is None and (prior_sampler is None or target_sampler is None):
            raise ValueError("Provide either a joint_sampler or both prior and target samplers.")

        self.model = model
        self.cond_path_fn = cond_path_fn
        self.cond_vec_field_fn = cond_vec_field_fn
        self.prior_sampler = prior_sampler
        self.target_sampler = target_sampler
        self.joint_sampler = joint_sampler
        self.ode_factory = ode_factory
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def update_model(self, new_model: nn.Module):
        self.model = new_model

    def _draw_joint(self, batch_size: int):
        if self.joint_sampler is not None:
            return self.joint_sampler.sample(batch_size)
        x0 = self.prior_sampler.sample(batch_size)
        x1 = self.target_sampler.sample(batch_size)
        return x0, x1

    def _draw_prior(self, batch_size: int):
        return self.prior_sampler.sample(batch_size)

    def _draw_target(self, batch_size: int):
        return self.target_sampler.sample(batch_size)

    def draw_samples(self, batch_size: int, t: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Draw raw samples (x0, x1, t). Use this to share the same batch across multiple bundles.
        """
        x0, x1 = self._draw_joint(batch_size)
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device)
        return {"x0": x0, "x1": x1, "t": t}

    # ------------------------------------------------------------------ #
    # Training helpers
    # ------------------------------------------------------------------ #
    def sample_batch(self, batch_size: int, shared_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Produce a training batch. If shared_data is provided, reuse its x0/x1/t so multiple
        models see the exact same inputs.
        """
        if shared_data is None:
            data = self.draw_samples(batch_size)
        else:
            data = dict(shared_data)  # shallow copy to avoid mutating caller state
            if "t" not in data:
                data["t"] = torch.rand(data["x0"].shape[0], device=data["x0"].device)
            if "x0" not in data:
                data["x0"] = self._draw_prior(data["t"].shape[0])
            if "x1" not in data:
                data["x1"] = self._draw_target(data["t"].shape[0])
        x0, x1, t = data["x0"], data["x1"], data["t"]
        xt = data.get("xt")
        if xt is None:
            xt = self.cond_path_fn(x0, x1, t)
        ut = data.get("ut")
        if ut is None:
            ut = self.cond_vec_field_fn(x0, x1, t)
        xt_in = torch.cat([xt, t[:, None]], dim=-1)
        vt = self.model(xt_in)
        data.update({"xt": xt, "ut": ut, "vt": vt})
        return data

    def loss(self, batch_size: int, loss_fn: Callable = nn.MSELoss(), shared_data: Optional[Dict[str, torch.Tensor]] = None):
        data = self.sample_batch(batch_size, shared_data=shared_data)
        return loss_fn(data["vt"], data["ut"])

    # ------------------------------------------------------------------ #
    # ODE helpers
    # ------------------------------------------------------------------ #
    def _require_ode(self):
        if self.ode_factory is None:
            raise RuntimeError("ode_factory was not provided; cannot build NeuralODE.")
        return self.ode_factory(self.model)

    @torch.no_grad()
    def forward_map(self, batch_size: int = None, inputs: Optional[torch.Tensor] = None, flatten: bool = True):
        """
        Push prior samples (or provided inputs) forward through the learned flow.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples to draw from the prior when ``inputs`` is None.
        inputs : Tensor, optional
            Explicit starting points shaped (batch, dim). When provided, no sampling occurs.
        flatten : bool
            If True, return data reshaped to (batch, -1).
        """
        if inputs is None:
            if batch_size is None:
                raise ValueError("Provide inputs or batch_size to forward_map.")
            x0 = self._draw_prior(batch_size)
        else:
            x0 = torch.as_tensor(inputs, device=self.device)
            batch_size = x0.shape[0]
        ode = self._require_ode()
        t_span = torch.linspace(0, 1, 2, device=x0.device)
        traj = ode.trajectory(x0, t_span=t_span)
        traj = traj.transpose(0, 1) if traj.dim() == x0.dim() + 1 else traj
        end = traj[-1] if traj.shape[0] == 2 else traj[:, -1]
        return end.reshape(batch_size, -1) if flatten else end

    @torch.no_grad()
    def backward_map(self, batch_size: int = None, inputs: Optional[torch.Tensor] = None, flatten: bool = True):
        """
        Pull target samples (or provided inputs) backward to the prior through the learned flow.
        """
        if inputs is None:
            if batch_size is None:
                raise ValueError("Provide inputs or batch_size to backward_map.")
            x1 = self._draw_target(batch_size)
        else:
            x1 = torch.as_tensor(inputs, device=self.device)
            batch_size = x1.shape[0]
        ode = self._require_ode()
        t_span = torch.linspace(1, 0, 2, device=x1.device)
        traj = ode.trajectory(x1, t_span=t_span)
        traj = traj.transpose(0, 1) if traj.dim() == x1.dim() + 1 else traj
        end = traj[-1] if traj.shape[0] == 2 else traj[:, -1]
        return end.reshape(batch_size, -1) if flatten else end
