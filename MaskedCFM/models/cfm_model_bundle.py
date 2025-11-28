import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any


__all__ = ["build_bundle","CFMModelBundle", "ModelSpec", "serialize_model_specs"]


def _callable_to_dict(fn: Callable) -> Dict[str, str]:
    return {
        "module": fn.__module__,
        "qualname": getattr(fn, "__qualname__", fn.__name__),
        "name": getattr(fn, "__name__", repr(fn)),
    }


def _class_to_dict(obj: Any, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = {
        "module": obj.__class__.__module__,
        "class": obj.__class__.__name__,
    }
    if extra:
        data.update(extra)
    return data


def _serialize_sampler(sampler) -> Dict[str, Any]:
    payload = _class_to_dict(sampler)
    if hasattr(sampler, "state_dict"):
        payload["state"] = sampler.state_dict()
    return payload


@dataclass
class ModelSpec:
    name: str
    long_name: str
    bundle: 'CFMModelBundle'
    train_history: List[Dict[str, Any]]
    val_history: List[Dict[str, Any]]
    checkpoints: List[Dict[str, Any]]

def build_bundle(name: str, long_name: str, bundle: 'CFMModelBundle'):
    return ModelSpec(name=name, long_name=long_name, bundle=bundle, train_history=[],  val_history=[],  checkpoints = [])


def serialize_model_specs(specs: List[ModelSpec]) -> List[Dict[str, Any]]:
    serialized = []
    for spec in specs:
        serialized.append({
            "name": spec.name,
            "long_name": spec.long_name,
            "bundle": spec.bundle.to_serializable(),
            "train_history": spec.train_history,
            "val_history": spec.val_history,
            "checkpoints": spec.checkpoints,
        })
    return serialized

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
        loss_fn: Callable = nn.MSELoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
        device="cpu",
        model_config: Optional[Dict[str, Any]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        ode_factory_config: Optional[Dict[str, Any]] = None,
        cond_path_config: Optional[Dict[str, Any]] = None,
        cond_vec_config: Optional[Dict[str, Any]] = None,
        sampler_config: Optional[Dict[str, Any]] = None,
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
        self.loss_fn = loss_fn
        self.training = True
        self.model_config = model_config or self._infer_model_config(model)
        if optimizer is not None:
            self.optimizer = optimizer
            self.optimizer_config = optimizer_config or self._infer_optimizer_config(optimizer)
        elif optimizer_factory is not None:
            self.optimizer = optimizer_factory(self.model)
            self.optimizer_config = optimizer_config or self._infer_optimizer_config(self.optimizer)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            self.optimizer_config = optimizer_config or self._infer_optimizer_config(self.optimizer)

        self.loss_config = loss_config or self._infer_loss_config(loss_fn)
        self.ode_factory_config = ode_factory_config or ({"name": getattr(ode_factory, "__name__", None)} if ode_factory else None)
        self.cond_path_config = cond_path_config or _callable_to_dict(cond_path_fn)
        self.cond_vec_config = cond_vec_config or _callable_to_dict(cond_vec_field_fn)
        self.sampler_config = sampler_config or self._infer_sampler_config()

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #
    def update_model(self, new_model: nn.Module):
        self.model = new_model

    def train(self, mode: bool = True):
        self.training = mode
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

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

    def loss(self, batch_size: int, shared_data: Optional[Dict[str, torch.Tensor]] = None):
        data = self.sample_batch(batch_size, shared_data=shared_data)
        loss_val = self.loss_fn(data["vt"], data["ut"])
        if self.optimizer is not None and self.training:
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()
        return loss_val

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

    # ------------------------------------------------------------------ #
    # Configuration helpers
    # ------------------------------------------------------------------ #
    def _infer_model_config(self, model: nn.Module) -> Dict[str, Any]:
        if hasattr(model, "to_config"):
            return model.to_config()
        return _class_to_dict(model, {"kwargs": {}})

    def _infer_loss_config(self, loss_fn: nn.Module) -> Dict[str, Any]:
        config = _class_to_dict(loss_fn)
        if hasattr(loss_fn, "state_dict"):
            config["state_dict"] = loss_fn.state_dict()
        params = {}
        if hasattr(loss_fn, "reduction"):
            params["reduction"] = loss_fn.reduction
        if params:
            config["kwargs"] = params
        return config

    def _infer_optimizer_config(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        config = _class_to_dict(optimizer)
        config["defaults"] = optimizer.defaults.copy()
        group_template = []
        for group in optimizer.param_groups:
            group_template.append({k: v for k, v in group.items() if k != "params"})
        config["param_groups"] = group_template
        return config

    def _infer_sampler_config(self) -> Dict[str, Any]:
        if self.joint_sampler is not None:
            return {"mode": "joint", "joint": _serialize_sampler(self.joint_sampler)}
        return {
            "mode": "independent",
            "prior": _serialize_sampler(self.prior_sampler),
            "target": _serialize_sampler(self.target_sampler),
        }

    def to_serializable(self) -> Dict[str, Any]:
        data = {
            "model": {
                "config": self.model_config,
                "state_dict": self.model.state_dict(),
            },
            "loss_fn": {
                "config": self.loss_config,
            },
            "optimizer": {
                "config": self.optimizer_config,
                "state_dict": self.optimizer.state_dict(),
            },
            "cond_path_fn": self.cond_path_config,
            "cond_vec_field_fn": self.cond_vec_config,
            "samplers": self.sampler_config,
            "ode_factory": self.ode_factory_config,
        }
        return data
