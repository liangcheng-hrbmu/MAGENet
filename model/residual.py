from __future__ import annotations
import math
from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F
from .components import _logit

class OrthogonalExpressionResidual(nn.Module):
    """Nonlinear residual with input- or representation-space separation."""

    def __init__(
        self,
        genes: int,
        *,
        knowledge_width: int,
        mode: str = "gm_null",
        dropout: float = 0.50,
        maximum_scale: float = 1.0,
        scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        if mode not in ("gm_null", "representation", "raw"):
            raise ValueError("residual mode must be 'gm_null', 'representation', or 'raw'")
        self.mode = str(mode)
        self.encoder = nn.Linear(int(genes), 128)
        self.project = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        for layer in (self.encoder, self.project, self.output):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.project.bias)
        generator = torch.Generator(device="cpu").manual_seed(
            9029 + 31 * int(knowledge_width)
        )
        knowledge_projection = torch.randn(
            int(knowledge_width), 32, generator=generator
        ) / math.sqrt(float(knowledge_width))
        self.register_buffer("knowledge_projection", knowledge_projection)
        self.maximum_scale = float(maximum_scale)
        self.scale_logit = nn.Parameter(torch.tensor(
            _logit(float(scale_init) / float(maximum_scale)), dtype=torch.float32
        ))

    def scale(self) -> torch.Tensor:
        return self.maximum_scale * torch.sigmoid(self.scale_logit)

    @staticmethod
    def orthogonal_basis(gm_coefficient: torch.Tensor) -> torch.Tensor:
        coefficient = gm_coefficient.detach()
        complete, _ = torch.linalg.qr(coefficient.double(), mode="complete")
        # Keep the conservative null envelope after all target columns.  If
        # GM is rank deficient this removes a few extra directions, but never
        # lets a GM-visible direction leak into the residual.
        return complete[:, coefficient.shape[1]:]

    def project_expression(
        self, expression: torch.Tensor, gm_coefficient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self.orthogonal_basis(gm_coefficient)
        projected = ((expression.double() @ basis) @ basis.T).to(expression)
        return projected, basis

    def forward(
        self,
        expression: torch.Tensor,
        gm_coefficient: torch.Tensor,
        *,
        knowledge_summary: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if self.mode == "gm_null":
            projected, basis = self.project_expression(expression, gm_coefficient)
        else:
            projected = expression
            basis = expression.new_zeros((expression.shape[1], 0), dtype=torch.float64)
        state = F.gelu(self.encoder(projected))
        state = self.dropout(F.gelu(self.project(self.dropout(state))))
        representation_error = state.new_zeros(())
        if self.mode == "representation":
            if knowledge_summary is None:
                raise ValueError("representation residual requires knowledge_summary")
            direction = knowledge_summary.detach() @ self.knowledge_projection.to(
                knowledge_summary
            )
            state64 = state.double()
            direction64 = direction.double()
            coefficient = (state64 * direction64).sum(1, keepdim=True) / (
                direction64.square().sum(1, keepdim=True) + 1e-12
            )
            state = (state64 - coefficient * direction64).to(state)
            state64 = state.double()
            coefficient = (state64 * direction64).sum(1, keepdim=True) / (
                direction64.square().sum(1, keepdim=True) + 1e-12
            )
            state = (state64 - coefficient * direction64).to(state)
            representation_error = (
                (state.double() * direction.double()).sum(1).abs()
                / (
                    state.double().norm(dim=1)
                    * direction.double().norm(dim=1)
                ).clamp_min(1e-12)
            ).max().to(state)
        raw = self.output(state).squeeze(1)
        logit = self.scale() * raw
        if not return_aux:
            return logit
        return logit, {
            "projected_expression": projected,
            "orthogonal_basis": basis,
            "raw_logit": raw,
            "scale": self.scale(),
            "mode": self.mode,
            "representation_relative_error": representation_error,
        }

    @torch.no_grad()
    def audit(
        self,
        expression: torch.Tensor,
        gm_coefficient: torch.Tensor,
        knowledge_summary: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if self.mode == "gm_null":
            projected, basis = self.project_expression(expression, gm_coefficient)
            cross = gm_coefficient.detach().double().T @ projected.double().T
            denominator = (
                gm_coefficient.detach().double().norm()
                * projected.double().norm()
            ).clamp_min(1e-12)
            complementarity_error = cross.norm() / denominator
        elif self.mode == "representation":
            _, auxiliary = self(
                expression,
                gm_coefficient,
                knowledge_summary=knowledge_summary,
                return_aux=True,
            )
            basis = auxiliary["orthogonal_basis"]
            complementarity_error = auxiliary["representation_relative_error"]
        else:
            basis = expression.new_zeros((expression.shape[1], 0), dtype=torch.float64)
            complementarity_error = expression.new_zeros(())
        return {
            "mode": self.mode,
            "complementarity_enforced": self.mode != "raw",
            "complementarity_error": float(complementarity_error.cpu()),
            "basis_orthonormal_max_error": (
                float((basis.double().T @ basis.double() - torch.eye(
                    basis.shape[1], dtype=torch.float64, device=basis.device
                )).abs().max().cpu()) if basis.shape[1] else 0.0
            ),
            "residual_scale": float(self.scale().cpu()),
            "maximum_residual_scale": self.maximum_scale,
            "output_bias_present": self.output.bias is not None,
        }
