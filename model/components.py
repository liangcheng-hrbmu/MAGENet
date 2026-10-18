from __future__ import annotations
import math
import torch
from torch import nn
def binary_mask(mask):
    if hasattr(mask, "toarray"):
        mask = mask.toarray()
    result = (torch.as_tensor(mask, dtype=torch.float32) > 0).float()
    if result.ndim != 2:
        raise ValueError("Masks must be two-dimensional")
    return result


class ZeroAnchoredNodeEncoder(nn.Module):
    """Encode signed deviations with a bounded odd map and no cross-gene mixing."""

    def __init__(self, nodes: int, width: int):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(nodes, width))
        nn.init.normal_(self.scale, mean=1.0, std=0.10)

    def forward(self, standardized_state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(standardized_state[:, :, None] * self.scale[None, :, :])


class DetachedRunningRMS(nn.Module):
    """Per-node opportunity measured by equal-signal control forward passes."""

    def __init__(
        self,
        nodes: int,
        momentum: float = 0.01,
        epsilon: float = 1e-6,
        minimum_attenuation: float = 0.25,
        cumulative_warmup_updates: int = 32,
    ):
        super().__init__()
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)
        self.minimum_attenuation = float(minimum_attenuation)
        self.cumulative_warmup_updates = int(cumulative_warmup_updates)
        if not 0.0 < self.minimum_attenuation <= 1.0:
            raise ValueError("minimum_attenuation must be in (0, 1]")
        if self.cumulative_warmup_updates < 1:
            raise ValueError("cumulative_warmup_updates must be positive")
        self.register_buffer("running_mean_square", torch.ones(nodes))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def update(self, state: torch.Tensor) -> None:
        if state.ndim != 3 or state.shape[1] != len(self.running_mean_square):
            raise ValueError("Opportunity state must have shape [batch, nodes, width]")
        batch_mean_square = state.detach().square().mean(dim=(0, 2)).clamp_min(self.epsilon)
        updates = int(self.updates.item())
        if updates == 0:
            self.running_mean_square.copy_(batch_mean_square)
        elif updates < self.cumulative_warmup_updates:
            weight = 1.0 / (updates + 1.0)
            self.running_mean_square.mul_(1.0 - weight).add_(batch_mean_square * weight)
        else:
            self.running_mean_square.mul_(1.0 - self.momentum).add_(batch_mean_square * self.momentum)
        self.updates.add_(1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        attenuation = self.attenuation().detach()
        return state * attenuation[None, :, None]

    def rms(self) -> torch.Tensor:
        return self.running_mean_square.clamp_min(self.epsilon).sqrt()

    def attenuation(self) -> torch.Tensor:
        """Only reduce above-median opportunity; never amplify a low-opportunity node."""
        rms = self.rms()
        reference = rms.median().clamp_min(self.epsilon)
        return (reference / rms).clamp(min=self.minimum_attenuation, max=1.0)

    def audit(self) -> dict:
        rms = self.rms().detach()
        attenuation = self.attenuation().detach()
        return {
            "updates": int(self.updates.item()),
            "cumulative_warmup_updates": self.cumulative_warmup_updates,
            "reference_rms": float(rms.median().cpu()),
            "minimum_allowed_attenuation": self.minimum_attenuation,
            "minimum_attenuation": float(attenuation.min().cpu()),
            "maximum_attenuation": float(attenuation.max().cpu()),
        }


def _logit(probability: float) -> float:
    value = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    return math.log(value / (1.0 - value))


class ScratchDirectLinearReadout(nn.Module):
    """Capacity-preserving readout with exact per-microbe contributions."""

    def __init__(self, width: int, microbes: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(int(width), 1)
        self.output = nn.Linear(int(microbes), 1)
        nn.init.kaiming_normal_(self.node_projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.node_projection.bias)
        nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
        nn.init.zeros_(self.output.bias)

    def forward(self, microbe: torch.Tensor, return_aux: bool = False):
        node_score = self.node_projection(microbe).squeeze(-1)
        node_contribution = node_score * self.output.weight[0][None, :]
        logit = node_contribution.sum(1) + self.output.bias.reshape(())
        if not return_aux:
            return logit
        return logit, {
            "node_score": node_score,
            "node_contribution": node_contribution,
            "response_weight": self.output.weight[0],
        }
