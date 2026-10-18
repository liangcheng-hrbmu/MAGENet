"""Structural attention with source capacity and edge allocation."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .components import binary_mask


class FullQBudgetedRelation(nn.Module):
    """W0-free relation with separable source capacity and edge allocation."""

    def __init__(
        self,
        mask: Any,
        width: int,
        source_basis: Any,
        *,
        maximum_edge_ratio: float = 8.0,
        maximum_source_ratio: float = 4.0,
        target_opportunity_power: float = 0.0,
    ) -> None:
        super().__init__()
        support = binary_mask(mask)
        basis = torch.as_tensor(np.asarray(source_basis), dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] != support.shape[0] or basis.shape[1] < 1:
            raise ValueError("source_basis must be a nonempty Q with one row per source")
        if (support.sum(1) == 0).any() or (support.sum(0) == 0).any():
            raise ValueError("Every source and target must have binary support")
        orthogonal_error = (basis.double().T @ basis.double() - torch.eye(
            basis.shape[1], dtype=torch.float64
        )).abs().max()
        if float(orthogonal_error) > 1e-5:
            raise ValueError("source_basis must be orthonormal")
        if not 0.0 <= float(target_opportunity_power) <= 1.0:
            raise ValueError("target_opportunity_power must lie in [0, 1]")
        uniform_route = support / support.sum(1, keepdim=True)
        uniform_opportunity = uniform_route.sum(0)
        reference = uniform_opportunity.median().clamp_min(1e-12)
        target_weight = (reference / uniform_opportunity.clamp_min(1e-12)).pow(
            float(target_opportunity_power)
        ).clamp(0.25, 4.0)
        fixed_route = uniform_route * target_weight[None, :]
        fixed_route = fixed_route / fixed_route.sum(1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("mask", support)
        self.register_buffer("fixed_route", fixed_route)
        self.register_buffer("uniform_target_opportunity", uniform_opportunity)
        self.register_buffer("fixed_target_opportunity", fixed_route.sum(0))
        self.target_opportunity_power = float(target_opportunity_power)
        self.register_buffer("source_basis", basis)
        self.edge_logit = nn.Parameter(torch.zeros_like(support))
        self.source_logit = nn.Parameter(torch.zeros(support.shape[0]))
        self.value = nn.Linear(int(width), int(width), bias=False)
        nn.init.eye_(self.value.weight)
        self.maximum_edge_log = math.log(float(maximum_edge_ratio))
        self.maximum_source_log = 0.5 * math.log(float(maximum_source_ratio))

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # These two buffers are deterministic topology summaries added after
        # the first tuning checkpoints.  Injecting them keeps those checkpoints
        # loadable without relaxing strictness for learned state.
        for name in ("uniform_target_opportunity", "fixed_target_opportunity"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def projected_log_capacity(self, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return torch.zeros_like(self.source_logit)
        raw = torch.tanh(self.source_logit)
        projected = raw - self.source_basis @ (self.source_basis.T @ raw)
        scale = projected.abs().max().clamp_min(1.0)
        return self.maximum_source_log * projected / scale

    def source_capacity(self, enabled: bool = True) -> torch.Tensor:
        return torch.exp(self.projected_log_capacity(enabled))

    def allocation(self, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return self.fixed_route
        residual = self.maximum_edge_log * torch.tanh(self.edge_logit)
        route = self.fixed_route * torch.exp(residual) * self.mask
        return route / route.sum(1, keepdim=True).clamp_min(1e-12)

    def effective_coefficient(
        self,
        *,
        source_enabled: bool = True,
        edge_enabled: bool = True,
    ) -> torch.Tensor:
        return self.source_capacity(source_enabled)[:, None] * self.allocation(edge_enabled)

    def forward(
        self,
        source: torch.Tensor,
        *,
        source_enabled: bool = True,
        edge_enabled: bool = True,
        return_aux: bool = False,
    ):
        capacity = self.source_capacity(source_enabled)
        allocation = self.allocation(edge_enabled)
        values = self.value(source) * capacity[None, :, None]
        target = torch.einsum("bsd,st->btd", values, allocation)
        if not return_aux:
            return target
        return target, {
            "allocation": allocation,
            "source_capacity": capacity[None, :].expand(source.shape[0], -1),
            "log_source_capacity": self.projected_log_capacity(source_enabled),
            "effective_coefficient": capacity[:, None] * allocation,
            "source_enabled": bool(source_enabled),
            "edge_enabled": bool(edge_enabled),
        }

    def source_parameters(self) -> Sequence[nn.Parameter]:
        return (self.source_logit,)

    def edge_parameters(self) -> Sequence[nn.Parameter]:
        return (self.edge_logit,)

    def attention_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.source_parameters()) + tuple(self.edge_parameters())

    def regularization(self) -> torch.Tensor:
        route = self.allocation(True).clamp_min(1e-12)
        prior = self.fixed_route.clamp_min(1e-12)
        kl = (route * (route.log() - prior.log()) * self.mask).sum() / self.mask.shape[0]
        return kl + 0.01 * self.projected_log_capacity(True).square().mean()


class MultiHeadSignedBudgetAttentionRelation(FullQBudgetedRelation):
    """Channel-split signed attention with an L1 budget in every head."""

    def __init__(self, *args, heads: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        width = int(self.value.weight.shape[0])
        if int(heads) < 2 or width % int(heads) != 0:
            raise ValueError("width must be divisible by at least two attention heads")
        self.heads = int(heads)
        self.head_width = width // self.heads
        local_seed = (
            int(torch.initial_seed()) + 7919 + 101 * self.heads
            + 17 * self.mask.shape[0] + self.mask.shape[1]
        ) % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(local_seed)
        noise = torch.randn(
            self.heads,
            *self.mask.shape,
            generator=generator,
            dtype=self.edge_logit.dtype,
        ) * 0.01
        self.edge_logit = nn.Parameter(math.atanh(0.5) + noise)

    def head_signed_allocation(self, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return self.fixed_route[None, :, :].expand(self.heads, -1, -1)
        raw = torch.tanh(self.edge_logit) * self.mask[None, :, :]
        return raw / raw.abs().sum(2, keepdim=True).clamp_min(1e-12)

    def signed_allocation(self, enabled: bool = True) -> torch.Tensor:
        return self.head_signed_allocation(enabled).mean(0)

    def allocation(self, enabled: bool = True) -> torch.Tensor:
        return self.head_signed_allocation(enabled).abs().mean(0)

    def effective_coefficient(
        self,
        *,
        source_enabled: bool = True,
        edge_enabled: bool = True,
    ) -> torch.Tensor:
        capacity = self.source_capacity(source_enabled)
        return capacity[:, None] * self.signed_allocation(edge_enabled)

    def forward(
        self,
        source: torch.Tensor,
        *,
        source_enabled: bool = True,
        edge_enabled: bool = True,
        return_aux: bool = False,
    ):
        capacity = self.source_capacity(source_enabled)
        signed = self.head_signed_allocation(edge_enabled)
        values = self.value(source).reshape(
            source.shape[0], source.shape[1], self.heads, self.head_width
        )
        values = values * capacity[None, :, None, None]
        target = torch.einsum("bshd,hst->bthd", values, signed).reshape(
            source.shape[0], signed.shape[2], -1
        )
        if not return_aux:
            return target
        return target, {
            "allocation": signed.abs().mean(0),
            "signed_allocation": signed.mean(0),
            "head_signed_allocation": signed,
            "source_capacity": capacity[None, :].expand(source.shape[0], -1),
            "log_source_capacity": self.projected_log_capacity(source_enabled),
            "effective_coefficient": capacity[:, None] * signed.mean(0),
            "source_enabled": bool(source_enabled),
            "edge_enabled": bool(edge_enabled),
        }

    def regularization(self) -> torch.Tensor:
        route = self.head_signed_allocation(True).abs().clamp_min(1e-12)
        prior = self.fixed_route[None, :, :].clamp_min(1e-12)
        kl = (
            route * (route.log() - prior.log()) * self.mask[None, :, :]
        ).sum() / (self.heads * self.mask.shape[0])
        return kl + 0.01 * self.projected_log_capacity(True).square().mean()


class NodewiseDynamicMultiHeadSignedBudgetAttentionRelation(
    MultiHeadSignedBudgetAttentionRelation
):
    """Node-specific, sample-conditional capacity with per-sample Full-Q control.

    The propagated values and the values used to estimate capacity are passed
    separately.  This lets the model regularize message values with dropout
    while computing an interpretable capacity from the corresponding clean
    node state.  Capacity is projected away from every registered Full-Q
    nuisance direction for every sample and has a hard layer-wise ratio bound.
    """

    def __init__(
        self,
        *args,
        maximum_dynamic_source_ratio: float = 2.0,
        reference_epsilon: float = 1e-6,
        context_mode: str = "local_node",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if float(maximum_dynamic_source_ratio) <= 1.0:
            raise ValueError("maximum_dynamic_source_ratio must exceed one")
        if float(reference_epsilon) <= 0.0:
            raise ValueError("reference_epsilon must be positive")
        if str(context_mode) not in ("local_node", "global_mean"):
            raise ValueError("context_mode must be 'local_node' or 'global_mean'")

        sources = int(self.mask.shape[0])
        width = int(self.value.weight.shape[0])
        # This relation has no static source multiplier.  Removing the
        # inherited parameter prevents a disabled or redundant capacity term
        # from entering the optimizer through the base parameter group.
        del self.source_logit
        self.source_query = nn.Parameter(torch.zeros(sources, width))
        self.maximum_dynamic_source_ratio = float(maximum_dynamic_source_ratio)
        self.maximum_dynamic_source_log = 0.5 * math.log(
            self.maximum_dynamic_source_ratio
        )
        self.reference_epsilon = float(reference_epsilon)
        self.context_mode = str(context_mode)
        self.register_buffer("source_reference_mean_square", torch.ones(sources))
        self.register_buffer(
            "source_reference_updates", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "source_reference_frozen", torch.zeros((), dtype=torch.bool)
        )

        basis64 = self.source_basis.double()
        projector = (
            torch.eye(sources, dtype=torch.float64)
            - basis64 @ basis64.T
        )
        projection_linf_norm = projector.abs().sum(1).max().clamp_min(1e-12)
        self.register_buffer("projection_linf_norm", projection_linf_norm)

    def projected_log_capacity(self, enabled: bool = True) -> torch.Tensor:
        # A static reference coefficient is still needed by the expression
        # residual and structural accessors.  M0 defines that reference at
        # unit capacity; all learned capacity is sample conditional.
        return self.source_basis[:, 0].new_zeros(self.mask.shape[0])

    def source_capacity(self, enabled: bool = True) -> torch.Tensor:
        return self.source_basis[:, 0].new_ones(self.mask.shape[0])

    def reference_effective_coefficient(
        self, *, edge_enabled: bool = True
    ) -> torch.Tensor:
        return self.signed_allocation(edge_enabled)

    def effective_coefficient(
        self,
        *,
        source_enabled: bool = True,
        edge_enabled: bool = True,
    ) -> torch.Tensor:
        return self.reference_effective_coefficient(edge_enabled=edge_enabled)

    @torch.no_grad()
    def reset_source_reference(self) -> None:
        self.source_reference_mean_square.fill_(1.0)
        self.source_reference_updates.zero_()
        self.source_reference_frozen.zero_()

    @torch.no_grad()
    def update_source_reference(self, clean_source: torch.Tensor) -> None:
        if bool(self.source_reference_frozen.item()):
            raise RuntimeError("source reference is frozen")
        if clean_source.ndim != 3 or clean_source.shape[1:] != self.source_query.shape:
            raise ValueError("clean_source must have shape [batch, sources, width]")
        if not bool(torch.isfinite(clean_source).all()):
            raise ValueError("clean_source contains non-finite values")
        observations = int(self.source_reference_updates.item())
        batch = int(clean_source.shape[0])
        batch_mean_square = clean_source.detach().double().square().mean(
            dim=(0, 2)
        )
        if observations == 0:
            updated = batch_mean_square
        else:
            updated = (
                self.source_reference_mean_square.double() * float(observations)
                + batch_mean_square * float(batch)
            ) / float(observations + batch)
        self.source_reference_mean_square.copy_(
            updated.clamp_min(self.reference_epsilon).to(
                self.source_reference_mean_square
            )
        )
        self.source_reference_updates.add_(batch)

    @torch.no_grad()
    def freeze_source_reference(self) -> None:
        if int(self.source_reference_updates.item()) < 1:
            raise RuntimeError("source reference cannot be frozen before an update")
        self.source_reference_frozen.fill_(True)

    def source_reference_rms(self) -> torch.Tensor:
        return self.source_reference_mean_square.clamp_min(
            self.reference_epsilon
        ).sqrt()

    def normalized_capacity_source(
        self, clean_source: torch.Tensor
    ) -> torch.Tensor:
        if clean_source.ndim != 3 or clean_source.shape[1:] != self.source_query.shape:
            raise ValueError("clean_source must have shape [batch, sources, width]")
        return clean_source / (
            self.source_reference_rms().detach()[None, :, None]
            + self.reference_epsilon
        )

    def capacity_context(self, clean_source: torch.Tensor) -> torch.Tensor:
        """Return the RMS-calibrated state used to generate node capacities."""
        normalized = self.normalized_capacity_source(clean_source)
        if self.context_mode == "local_node":
            return normalized
        global_mean = normalized.mean(1)
        return F.layer_norm(global_mean, (global_mean.shape[-1],))

    def raw_source_score(self, clean_source: torch.Tensor) -> torch.Tensor:
        context = self.capacity_context(clean_source)
        if self.context_mode == "local_node":
            return torch.einsum("bsd,sd->bs", context, self.source_query)
        return torch.einsum("bd,sd->bs", context, self.source_query)

    def sample_residual_log_source_capacity(
        self, clean_source: torch.Tensor, enabled: bool = True
    ) -> torch.Tensor:
        if clean_source.ndim != 3 or clean_source.shape[1:] != self.source_query.shape:
            raise ValueError("clean_source must have shape [batch, sources, width]")
        if not enabled:
            return clean_source.new_zeros(clean_source.shape[:2])
        score = self.raw_source_score(clean_source)
        raw = torch.tanh(score / math.sqrt(float(clean_source.shape[-1])))

        # Perform the projection in float64 and remove round-off twice.  Q is
        # fold-local and fixed, so gradients only flow through the raw scores.
        basis64 = self.source_basis.double()
        projected64 = raw.double()
        for _ in range(2):
            projected64 = projected64 - (projected64 @ basis64) @ basis64.T
        projected64 = (
            self.maximum_dynamic_source_log
            * projected64
            / self.projection_linf_norm
        )
        for _ in range(2):
            projected64 = projected64 - (projected64 @ basis64) @ basis64.T
        return projected64.to(raw)

    def sample_log_source_capacity(
        self, clean_source: torch.Tensor, enabled: bool = True
    ) -> torch.Tensor:
        return self.sample_residual_log_source_capacity(clean_source, enabled)

    def configured_maximum_sample_capacity_ratio(self) -> float:
        return self.maximum_dynamic_source_ratio

    def sample_source_capacity(
        self, clean_source: torch.Tensor, enabled: bool = True
    ) -> torch.Tensor:
        if not enabled:
            return clean_source.new_ones(clean_source.shape[:2])
        return torch.exp(self.sample_log_source_capacity(clean_source, True))

    def forward(
        self,
        source: torch.Tensor,
        *,
        capacity_source: Optional[torch.Tensor] = None,
        source_enabled: bool = True,
        edge_enabled: bool = True,
        return_aux: bool = False,
    ):
        clean_source = source if capacity_source is None else capacity_source
        if clean_source.shape != source.shape:
            raise ValueError("capacity_source must match the propagated source state")
        capacity = self.sample_source_capacity(clean_source, source_enabled)
        signed = self.head_signed_allocation(edge_enabled)
        values = self.value(source).reshape(
            source.shape[0], source.shape[1], self.heads, self.head_width
        )
        values = values * capacity[:, :, None, None]
        target = torch.einsum("bshd,hst->bthd", values, signed).reshape(
            source.shape[0], signed.shape[2], -1
        )
        if not return_aux:
            return target
        return target, {
            "allocation": signed.abs().mean(0),
            "signed_allocation": signed.mean(0),
            "head_signed_allocation": signed,
            "source_capacity": capacity,
            "log_source_capacity": capacity.log(),
            "effective_coefficient": self.reference_effective_coefficient(
                edge_enabled=edge_enabled
            ),
            "sample_effective_coefficient_factorized": True,
            "capacity_source": clean_source,
            "source_enabled": bool(source_enabled),
            "edge_enabled": bool(edge_enabled),
            "sample_conditional_source": True,
        }

    def source_parameters(self) -> Sequence[nn.Parameter]:
        return (self.source_query,)


class HeadCapacityMixin:
    tie_heads = False

    def install_head_capacity(self):
        # Replace the source query, with no second scalar capacity in parallel.
        self.source_query = nn.Parameter(self.source_query.detach()[None].expand(self.heads, -1, -1).clone())

    def _check_source(self, clean):
        if clean.ndim != 3 or clean.shape[1:] != (self.mask.shape[0], self.value.weight.shape[1]):
            raise ValueError('Source must have shape [batch,sources,width]')

    @torch.no_grad()
    def update_source_reference(self, clean_source):
        if bool(self.source_reference_frozen.item()):
            raise RuntimeError('source reference is frozen')
        self._check_source(clean_source)
        if not torch.isfinite(clean_source).all():
            raise ValueError('Non-finite source')
        count, batch = int(self.source_reference_updates), len(clean_source)
        value = clean_source.detach().double().square().mean((0, 2))
        if count:
            value = (self.source_reference_mean_square.double() * count + value * batch) / (count + batch)
        self.source_reference_mean_square.copy_(value.clamp_min(self.reference_epsilon).to(self.source_reference_mean_square))
        self.source_reference_updates.add_(batch)

    def normalized_capacity_source(self, clean_source):
        self._check_source(clean_source)
        return clean_source / (self.source_reference_rms().detach()[None, :, None] + self.reference_epsilon)

    def raw_source_score(self, clean_source):
        assert self.context_mode == 'global_mean'
        return torch.einsum('bd,hsd->bhs', self.capacity_context(clean_source), self.source_query)

    def sample_residual_log_source_capacity(self, clean_source, enabled=True):
        self._check_source(clean_source)
        if not enabled:
            return clean_source.new_zeros(len(clean_source), self.heads, self.mask.shape[0])
        raw = torch.tanh(self.raw_source_score(clean_source) / math.sqrt(clean_source.shape[-1]))
        basis = self.source_basis.double()
        projected = raw.double()
        for _ in range(2):
            projected = projected - (projected @ basis) @ basis.T
        projected = self.maximum_dynamic_source_log * projected / self.projection_linf_norm
        for _ in range(2):
            projected = projected - (projected @ basis) @ basis.T
        if self.tie_heads:
            projected = projected.mean(1, keepdim=True).expand_as(projected)
        return projected.to(raw)

    def sample_source_capacity(self, clean_source, enabled=True):
        return self.sample_log_source_capacity(clean_source, enabled).exp()


class BalancedGateMixin:
    """Explicit node and supported-edge gates; no post-gate renormalization."""

    def install_balanced_gates(self):
        self.node_gate_logit = nn.Parameter(self.mask.new_zeros(self.mask.shape[0]))
        self.edge_gate_logit = nn.Parameter(self.mask.new_zeros(self.edge_logit.shape))

    def node_gate(self, enabled=True):
        if not enabled:
            return self.mask.new_ones(self.mask.shape[0])
        raw = 1.0 + 0.5 * torch.tanh(self.node_gate_logit)
        # Preserve the parent's Full-Q geometric-mean source invariant.
        return (raw.log() - raw.log().mean()).exp()

    def edge_gate(self, enabled=True):
        if not enabled:
            return self.mask.new_ones(self.edge_logit.shape)
        return 1.0 + 0.5 * torch.tanh(self.edge_gate_logit) * self.mask[None]

    def effective_source_capacity(self, clean_source, enabled=True):
        base = super().sample_source_capacity(clean_source, enabled)
        return base * self.node_gate(enabled)[None, None, :]

    def _gated_route(self, base, enabled=True):
        """Apply a bounded positive gate to each existing signed edge."""
        if not enabled:
            return base
        gate = self.edge_gate(True)
        return base * gate

    def effective_head_signed_allocation(self, enabled=True):
        return self._gated_route(super().head_signed_allocation(enabled), enabled)

    def reference_effective_coefficient(self, *, edge_enabled=True):
        # All coefficient accessors must expose the same gated route used
        # by forward(), so audits and interpretation cannot silently inspect a
        # pre-gate parent route.
        return self.effective_head_signed_allocation(edge_enabled)

    def forward(self, source, *, capacity_source=None, source_enabled=True,
                edge_enabled=True, return_aux=False):
        clean = source if capacity_source is None else capacity_source
        if clean.shape != source.shape:
            raise ValueError('capacity_source must match source')
        capacity = self.effective_source_capacity(clean, source_enabled)
        signed = self.effective_head_signed_allocation(edge_enabled)
        values = self.value(source).reshape(len(source), source.shape[1], self.heads, self.head_width)
        values = values * capacity.transpose(1, 2)[:, :, :, None]
        target = torch.einsum('bshd,hst->bthd', values, signed).reshape(
            len(source), signed.shape[2], -1)
        if not return_aux:
            return target
        return target, dict(
            allocation=signed.abs().mean(0), signed_allocation=signed.mean(0),
            head_signed_allocation=signed, source_capacity=capacity,
            log_source_capacity=capacity.clamp_min(1e-12).log(),
            source_capacity_axes='batch,head,source',
            effective_coefficient=self.reference_effective_coefficient(edge_enabled=edge_enabled),
            sample_effective_coefficient_factorized=True,
            capacity_source=clean, source_enabled=bool(source_enabled),
            edge_enabled=bool(edge_enabled), sample_conditional_source=True,
            node_gate=self.node_gate(source_enabled), edge_gate=self.edge_gate(edge_enabled),
            route_budget_preserving=False)

    def source_parameters(self):
        return tuple(super().source_parameters()) + (self.node_gate_logit,)

    def edge_parameters(self):
        return tuple(super().edge_parameters()) + (self.edge_gate_logit,)

    @torch.no_grad()
    def balanced_audit(self):
        node = self.node_gate(True)
        edge = self.edge_gate(True)
        support = self.mask.bool()[None].expand_as(edge)
        return dict(
            node_gate_min=float(node.min()), node_gate_max=float(node.max()),
            node_gate_geometric_mean=float(node.double().log().mean().exp()),
            edge_gate_min=float(edge[support].min()), edge_gate_max=float(edge[support].max()),
            unsupported_edge_gate_max_abs=float(((edge - 1) * (~support)).abs().max()),
            neutral_node_gate_error=float((self.node_gate(False) - 1).abs().max()),
            neutral_edge_gate_error=float((self.edge_gate(False) - 1).abs().max()),
            node_gate_parameters=int(self.node_gate_logit.numel()),
            edge_gate_parameters=int(self.edge_gate_logit.numel()),
            route_normalize_dim=int(self.route_normalize_dim))


class FreeGeneMetaboliteRelation(NodewiseDynamicMultiHeadSignedBudgetAttentionRelation):
    def head_signed_allocation(self, enabled=True):
        return self.edge_logit * self.mask[None]

    def edge_parameters(self):
        # The free GM propagation weights belong to the base optimizer group.
        return ()

    def regularization(self):
        return self.edge_logit.new_zeros(())


class ReceiverMetaboliteMicrobeRelation(NodewiseDynamicMultiHeadSignedBudgetAttentionRelation):
    def receiver_components(self):
        raw = self.edge_logit.tanh() * self.mask[None]
        real_mass = raw.abs().sum(1)
        log_null = self.null_log_prior + self.receiver_null_logit
        denominator = torch.logaddexp(real_mass.clamp_min(1e-30).log(), log_null[None])
        signed = 2 * raw * torch.exp(-denominator)[:, None, :]
        null = torch.exp(log_null[None] - denominator)
        return signed, null

    def head_signed_allocation(self, enabled=True):
        if not enabled:
            return self.receiver_prior[None].expand(self.heads, -1, -1)
        return self.receiver_components()[0]

    def edge_parameters(self):
        return self.edge_logit, self.receiver_null_logit

    def regularization(self):
        return self.edge_logit.new_zeros(())


class GeneMetaboliteRelation(BalancedGateMixin, HeadCapacityMixin, FreeGeneMetaboliteRelation):
    route_normalize_dim = 2  # Source/target axis metadata; no final normalization.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.edge_logit.copy_(self.fixed_route[None].expand_as(self.edge_logit))
        self.install_head_capacity()
        self.install_balanced_gates()


class MetaboliteMicrobeRelation(BalancedGateMixin, HeadCapacityMixin, ReceiverMetaboliteMicrobeRelation):
    route_normalize_dim = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        degree = self.mask.sum(0)
        self.register_buffer("receiver_degree", degree)
        self.register_buffer("receiver_prior", self.mask / degree.clamp_min(1)[None])
        self.register_buffer("null_log_prior", (.5 * degree.clamp_min(1)).log())
        self.receiver_null_logit = nn.Parameter(self.mask.new_zeros(self.mask.shape[1]))
        self.install_head_capacity()
        self.install_balanced_gates()
