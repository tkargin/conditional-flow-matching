import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

__all__ = ["BlockLowerTriangularMask", "MaskedLinear", "MaskedBlockMLP"]

def BlockLowerTriangularMask(in_T, out_sizes, shared=None, device=None):
    """Create a block lower-triangular mask for MaskedLinear layer.
    Args:
        in_T: tuple of input block sizes (e.g. (d_i, d_i, ..., d_i))
        out_sizes: tuple of output block sizes (e.g. (d_o, d_o, ..., d_o))
        shared: list of sizes for shared blocks (always connected)
        device: device for the mask tensor
    Returns:
        A mask tensor of shape (sum(out_sizes), sum(in_sizes))
    """
    # in_sizes e.g. [d_i, d_i, ..., d_i, 1]  (time block last)
    shared = shared or []
    in_sizes = list(in_T) + list(shared)
    shared_start = len(in_T)

    rows = []
    for row_idx, out_dim in enumerate(out_sizes):
        allowed = set(range(row_idx + 1))
        allowed |= set(range(shared_start, len(in_sizes)))  # shared always on
        row = []
        for col_idx, in_dim in enumerate(in_sizes):
            val = 1.0 if col_idx in allowed else 0.0
            row.append(torch.full((out_dim, in_dim), val, device=device))
        rows.append(torch.cat(row, dim=1))
    return torch.cat(rows, dim=0)

# Masked Linear layer
class MaskParam(nn.Module):
    '''Parameterization that applies a mask to the weights.'''
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, weight):
        return weight * self.mask

class MaskedLinear(nn.Linear):
    '''Linear layer with a configurable mask on the weights.'''

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))
        parametrize.register_parametrization(self, "weight", MaskParam(self.mask))

    def set_mask(self, mask: torch.Tensor):
        if mask.shape != self.mask.shape:
            raise ValueError("Mask shape mismatch.")
        self.mask.copy_(mask)
        self.parametrizations.weight[0].mask.copy_(mask)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        #return nn.functional.linear(x, self.weight * self.mask, self.bias)
        return nn.functional.linear(x, self.weight, self.bias)



# Masked MLP with block lower-triangular masking
class MaskedBlockMLP(nn.Module):
    '''Masked MLP with block lower-triangular masking.
    Args:
        T: number of time steps
        in_dim: input dimension per time step
        out_dim: output dimension per time step
        hidden_per_t: tuple of hidden layer sizes per time step
        activation: activation function class (e.g. nn.ReLU)
        causal: if True, apply block lower-triangular masking
        debugMode: if True, print debug information
        time_varying: if True, include time feature in input
        flatten_output: if True, flatten output to (B, T*out_dim)
    '''
    def __init__(self, T, in_dim, out_dim, hidden_per_t=(64,64), 
                 activation=nn.SELU, causal=False, debugMode=False, time_varying=False, flatten_output=True):
        super().__init__()
        self.T = T
        self.in_dim = in_dim
        self.debugMode = debugMode
        self.time_varying = time_varying
        self.flatten_output = flatten_output
        
        hidden = list(hidden_per_t)
        layer_in_blocks = [[in_dim] * T]
        first_shared = [1] if time_varying else [] # shared block is appended only for the input layer
        for width in hidden:
            layer_in_blocks.append([width] * T)
        layer_out_blocks = [[width] * T for width in hidden] + [[out_dim] * T]

        layers = []
        self._block_layouts = []
        for idx, (in_blocks, out_blocks) in enumerate(zip(layer_in_blocks, layer_out_blocks)):
            shared_block = first_shared if (idx == 0 and time_varying) else []
            linear = MaskedLinear(sum(in_blocks) + sum(shared_block), sum(out_blocks))
            if causal:
                mask = BlockLowerTriangularMask(
                    in_blocks, out_blocks, shared=shared_block, device=linear.weight.device
                )
                linear.set_mask(mask)
            layers.append(linear)
            self._block_layouts.append({
                "in_blocks": list(in_blocks),
                "shared_block": list(shared_block),
                "out_blocks": list(out_blocks),
            })
        
        self.layers = nn.ModuleList(layers)
        self.act = activation() if activation is not None else nn.Identity()
    
    def forward(self, xt):
        # Accept (B, T, in_dim) or flattened (B, T*in_dim [+ time])
        if xt.dim() == 3:
            xt = xt.reshape(xt.size(0), -1)

        expected = self.T * self.in_dim + (1 if self.time_varying else 0)
        if xt.shape[-1] != expected:
            raise ValueError(f"Expected {expected} features, got {xt.shape[-1]}.")

        if self.time_varying:
            x_flat, t_feat = xt[:, : self.T * self.in_dim], xt[:, -1:]
        else:
            x_flat, t_feat = xt, None

        h = torch.cat([x_flat, t_feat], dim=-1) if t_feat is not None else x_flat

        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        y = self.layers[-1](h)

        if self.flatten_output:
            return y  # shape (B, T*out_dim) → identical to original MLP interface
        return y.view(xt.size(0), self.T, -1)

    def block_weight_magnitudes(self, norm="fro", include_shared=False):
        """Return per-block weight magnitudes for each masked layer."""
        magnitudes = []
        for layer, layout in zip(self.layers, self._block_layouts):
            in_blocks = list(layout["in_blocks"])
            if include_shared and layout["shared_block"]:
                in_blocks = in_blocks + list(layout["shared_block"])
            out_blocks = list(layout["out_blocks"])
            weight = layer.weight.detach()
            block_vals = torch.zeros(len(out_blocks), len(in_blocks))
            row_start = 0
            for r, out_dim in enumerate(out_blocks):
                col_start = 0
                for c, in_dim in enumerate(in_blocks):
                    block = weight[row_start:row_start + out_dim, col_start:col_start + in_dim]
                    if norm == "fro":
                        val = torch.linalg.norm(block, ord="fro")
                    elif norm == "absmax":
                        val = block.abs().max()
                    elif norm == "l1":
                        val = block.abs().sum()
                    else:
                        raise ValueError(f"Unsupported norm '{norm}'.")
                    block_vals[r, c] = val
                    col_start += in_dim
                row_start += out_dim
            magnitudes.append(block_vals.cpu().numpy())
        return magnitudes
