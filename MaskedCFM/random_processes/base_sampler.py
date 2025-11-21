import abc
import torch

class Sampler(abc.ABC):
    """Abstract base class for all samplers/generators."""
    def __init__(self, name, **config):
        self.name = name
        self.config = config

    @abc.abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    def state_dict(self):
        return {"name": self.name, "config": self.config}

    @classmethod
    @abc.abstractmethod
    def from_state_dict(cls, state):
        raise NotImplementedError


__all__ = ["Sampler"]
