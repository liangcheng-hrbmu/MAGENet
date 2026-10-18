from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence
import torch
from torch import nn
from .components import ZeroAnchoredNodeEncoder, DetachedRunningRMS, ScratchDirectLinearReadout
from .attention import NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
from .attention import GeneMetaboliteRelation, MetaboliteMicrobeRelation
from .context import build_topology_peer_matrix, TopologyPeerVirtualContext
from .residual import OrthogonalExpressionResidual

class MAGENet(nn.Module):
    """MAGENet: knowledge propagation, metabolite context and residual."""

    def __init__(
        self,
        graph: Any,
        gm_source_basis: Any,
        mm_source_basis: Any,
        *,
        width: int = 32,
        attention_enabled: bool = True,
        source_enabled: bool = True,
        edge_enabled: bool = True,
        virtual_enabled: bool = True,
        residual_enabled: bool = True,
        dropout: float = 0.10,
        residual_dropout: float = 0.65,
        gm_maximum_edge_ratio: float = 4.0,
        mm_maximum_edge_ratio: float = 8.0,
        maximum_source_ratio: float = 1.0,
        minimum_opportunity_attenuation: float = 0.75,
        peer_top_k: int = 4,
        maximum_virtual_scale: float = 0.20,
        virtual_scale_init: float = 0.05,
        maximum_residual_scale: float = 1.0,
        residual_scale_init: float = 0.10,
        residual_mode: str = "representation",
        shuffle_virtual_peers: bool = False,
        peer_seed: int = 1992,
        relation_kind: str = "nodewise_dynamic_multihead_signed_attention",
        relation_heads: int = 16,
        maximum_head_gate_ratio: float = 4.0,
        maximum_dynamic_source_ratio: float = 8.0,
        maximum_fixed_source_ratio: float = 1.5,
        dynamic_source_context_mode: str = "global_mean",
        target_opportunity_power: float = 0.0,
        gm_target_opportunity_power: Optional[float] = None,
        mm_target_opportunity_power: Optional[float] = None,
        virtual_clean_mode: str = "orthogonal_innovation",
        virtual_targetwise_scale: bool = True,
        virtual_samplewise_gate: bool = True,
        virtual_samplewise_gate_mode: str = "unbounded_logit",
        virtual_maximum_clean_gate_ratio: float = 4.0,
        virtual_learnable_peers: bool = False,
        virtual_maximum_peer_ratio: float = 4.0,
        virtual_samplewise_peer_attention: bool = True,
        virtual_peer_attention_strength_init: float = 0.25,
        virtual_peer_attention_query_mode: str = "target",
        virtual_mm_capacity_coupling: float = 0.0,
        normalize_encoder_scale: bool = False,
        control_rms_neutral: bool = True,
        residual_fusion_weight: float = 1.0,
        pool_kind: str = "direct",
        residual_variance_balance: bool = False,
    ) -> None:
        super().__init__()
        genes, metabolites, microbes, gm_mask, mm_mask = graph.as_tuple()
        self.graph = graph
        self.nodes = {
            "genes": len(genes),
            "metabolites": len(metabolites),
            "microbes": len(microbes),
        }
        self.width = int(width)
        self.normalize_encoder_scale = bool(normalize_encoder_scale)
        self.control_rms_neutral = bool(control_rms_neutral)
        self.residual_fusion_weight = float(residual_fusion_weight)
        if not 0.0 < self.residual_fusion_weight <= 1.0:
            raise ValueError("residual_fusion_weight must lie in (0, 1]")
        self.attention_enabled = bool(attention_enabled)
        self.relation_heads = int(relation_heads)
        self.pool_kind = str(pool_kind)
        self.residual_variance_balance = bool(residual_variance_balance)
        if self.residual_variance_balance:
            self.register_buffer("knowledge_logit_variance", torch.ones(()))
            self.register_buffer("residual_logit_variance", torch.ones(()))
            self.register_buffer(
                "residual_balance_updates", torch.zeros((), dtype=torch.long)
            )
        self.source_enabled = bool(source_enabled and attention_enabled)
        self.edge_enabled = bool(edge_enabled and attention_enabled)
        self.virtual_enabled = bool(virtual_enabled)
        self.virtual_mm_capacity_coupling = float(virtual_mm_capacity_coupling)
        if not 0.0 <= self.virtual_mm_capacity_coupling <= 1.0:
            raise ValueError("virtual_mm_capacity_coupling must lie in [0, 1]")
        self.residual_enabled = bool(residual_enabled)
        gm_target_power = (
            float(target_opportunity_power)
            if gm_target_opportunity_power is None
            else float(gm_target_opportunity_power)
        )
        mm_target_power = (
            float(target_opportunity_power)
            if mm_target_opportunity_power is None
            else float(mm_target_opportunity_power)
        )
        self.encoder = ZeroAnchoredNodeEncoder(len(genes), self.width)
        if relation_kind != "nodewise_dynamic_multihead_signed_attention":
            raise ValueError("This package implements the nodewise relation only")
        if dynamic_source_context_mode != "global_mean":
            raise ValueError("MAGENet uses global_mean capacity context")
        if residual_mode != "representation" or pool_kind != "direct":
            raise ValueError("MAGENet uses a representation residual and direct readout")
        self.relation_kind = str(relation_kind)
        gm_relation_extra = mm_relation_extra = dict(
            heads=int(relation_heads),
            maximum_dynamic_source_ratio=float(maximum_dynamic_source_ratio),
            context_mode=str(dynamic_source_context_mode),
        )
        self.gm = GeneMetaboliteRelation(
            gm_mask,
            self.width,
            gm_source_basis,
            maximum_edge_ratio=float(gm_maximum_edge_ratio),
            maximum_source_ratio=float(maximum_source_ratio),
            target_opportunity_power=gm_target_power,
            **gm_relation_extra,
        )
        self.mm = MetaboliteMicrobeRelation(
            mm_mask,
            self.width,
            mm_source_basis,
            maximum_edge_ratio=float(mm_maximum_edge_ratio),
            maximum_source_ratio=float(maximum_source_ratio),
            target_opportunity_power=mm_target_power,
            **mm_relation_extra,
        )
        self.metabolite_opportunity = DetachedRunningRMS(
            len(metabolites), minimum_attenuation=float(minimum_opportunity_attenuation)
        )
        self.microbe_opportunity = DetachedRunningRMS(
            len(microbes), minimum_attenuation=float(minimum_opportunity_attenuation)
        )
        peers, peer_metadata = build_topology_peer_matrix(
            gm_mask,
            mm_mask,
            top_k=int(peer_top_k),
            shuffle=bool(shuffle_virtual_peers),
            seed=int(peer_seed),
        )
        self.peer_metadata = peer_metadata
        self.virtual_context = TopologyPeerVirtualContext(
            peers,
            self.width,
            maximum_clean_scale=float(maximum_virtual_scale),
            clean_scale_init=float(virtual_scale_init),
            clean_mode=str(virtual_clean_mode),
            targetwise_clean_scale=bool(virtual_targetwise_scale),
            samplewise_clean_gate=bool(virtual_samplewise_gate),
            samplewise_clean_gate_mode=str(virtual_samplewise_gate_mode),
            maximum_clean_gate_ratio=float(virtual_maximum_clean_gate_ratio),
            learnable_peer_allocation=bool(virtual_learnable_peers),
            maximum_peer_ratio=float(virtual_maximum_peer_ratio),
            samplewise_peer_attention=bool(virtual_samplewise_peer_attention),
            peer_attention_strength_init=float(
                virtual_peer_attention_strength_init
            ),
            peer_attention_query_mode=str(virtual_peer_attention_query_mode),
        )
        self.pool = ScratchDirectLinearReadout(self.width, len(microbes))
        self.residual = OrthogonalExpressionResidual(
            len(genes),
            knowledge_width=self.width,
            mode=str(residual_mode),
            dropout=float(residual_dropout),
            maximum_scale=float(maximum_residual_scale),
            scale_init=float(residual_scale_init),
        )
        self.dropout = nn.Dropout(float(dropout))

    def source_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.gm.source_parameters()) + tuple(self.mm.source_parameters())

    def edge_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.gm.edge_parameters()) + tuple(self.mm.edge_parameters())

    def attention_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.source_parameters()) + tuple(self.edge_parameters())

    def virtual_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.virtual_context.parameters())

    def residual_parameters(self) -> Sequence[nn.Parameter]:
        return tuple(self.residual.parameters())

    def trainable_parameters(self) -> Sequence[nn.Parameter]:
        disabled: set[int] = set()
        if not self.source_enabled:
            disabled.update(id(value) for value in self.source_parameters())
        if not self.edge_enabled:
            disabled.update(id(value) for value in self.edge_parameters())
        if not self.virtual_enabled:
            disabled.update(id(value) for value in self.virtual_parameters())
        if not self.residual_enabled:
            disabled.update(id(value) for value in self.residual_parameters())
        return [value for value in self.parameters() if id(value) not in disabled]

    def parameter_groups(
        self,
        base_learning_rate: float,
        *,
        source_learning_rate: float,
        edge_learning_rate: float,
        virtual_learning_rate: float,
        residual_learning_rate: float,
    ) -> list[Dict[str, Any]]:
        selected = {id(value) for value in self.trainable_parameters()}
        categories = {
            "source": {id(value) for value in self.source_parameters()} & selected,
            "edge": {id(value) for value in self.edge_parameters()} & selected,
            "virtual": {id(value) for value in self.virtual_parameters()} & selected,
            "residual": {id(value) for value in self.residual_parameters()} & selected,
        }
        categories["base"] = selected - set().union(*categories.values())
        rates = {
            "base": float(base_learning_rate),
            "source": float(source_learning_rate),
            "edge": float(edge_learning_rate),
            "virtual": float(virtual_learning_rate),
            "residual": float(residual_learning_rate),
        }
        groups = []
        for name in ("base", "source", "edge", "virtual", "residual"):
            values = [value for value in self.parameters() if id(value) in categories[name]]
            if values:
                groups.append({"name": name, "params": values, "lr": rates[name]})
        return groups

    def encode_gene(self, expression: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        values = expression.clamp(-5.0, 5.0)
        if self.normalize_encoder_scale:
            scale = self.encoder.scale
            scale = scale / scale.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            state = torch.tanh(values[:, :, None] * scale[None, :, :])
        else:
            state = self.encoder(values)
        return self.dropout(state) if apply_dropout else state

    def _metabolite(
        self,
        gene: torch.Tensor,
        *,
        capacity_source: Optional[torch.Tensor] = None,
        source_enabled: bool,
        edge_enabled: bool,
        return_aux: bool = False,
    ):
        options = {
            "source_enabled": source_enabled,
            "edge_enabled": edge_enabled,
            "return_aux": True,
        }
        if isinstance(
            self.gm, NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
        ):
            options["capacity_source"] = (
                gene if capacity_source is None else capacity_source
            )
        raw, auxiliary = self.gm(gene, **options)
        state = self.dropout(torch.tanh(self.metabolite_opportunity(raw)))
        return (state, raw, auxiliary) if return_aux else state

    def _microbe(
        self,
        metabolite: torch.Tensor,
        *,
        capacity_source: Optional[torch.Tensor] = None,
        source_enabled: bool,
        edge_enabled: bool,
        return_aux: bool = False,
    ):
        options = {
            "source_enabled": source_enabled,
            "edge_enabled": edge_enabled,
            "return_aux": True,
        }
        if isinstance(self.mm, (
            NodewiseDynamicMultiHeadSignedBudgetAttentionRelation,
        )):
            options["capacity_source"] = (
                metabolite if capacity_source is None else capacity_source
            )
        raw, auxiliary = self.mm(metabolite, **options)
        state = self.dropout(torch.tanh(self.microbe_opportunity(raw)))
        return (state, raw, auxiliary) if return_aux else state

    def _configured_gm_coefficient(self) -> torch.Tensor:
        # Knockouts must not silently change what information the residual sees.
        return self.gm.effective_coefficient(
            source_enabled=self.source_enabled,
            edge_enabled=self.edge_enabled,
        )

    @torch.no_grad()
    def _update_residual_balance(
        self, knowledge_logit: torch.Tensor, residual_logit: torch.Tensor
    ) -> None:
        if not self.residual_variance_balance:
            return
        knowledge_variance = knowledge_logit.detach().var(unbiased=False).clamp_min(1e-6)
        residual_variance = residual_logit.detach().var(unbiased=False).clamp_min(1e-6)
        updates = int(self.residual_balance_updates.item())
        if updates == 0:
            weight = 1.0
        elif updates < 32:
            weight = 1.0 / float(updates + 1)
        else:
            weight = 0.01
        self.knowledge_logit_variance.mul_(1.0 - weight).add_(
            knowledge_variance * weight
        )
        self.residual_logit_variance.mul_(1.0 - weight).add_(
            residual_variance * weight
        )
        self.residual_balance_updates.add_(1)

    def residual_balance_factor(self) -> torch.Tensor:
        if not self.residual_variance_balance:
            return self.residual.scale().new_ones(())
        return torch.sqrt(
            self.knowledge_logit_variance.clamp_min(1e-6)
            / self.residual_logit_variance.clamp_min(1e-6)
        ).clamp(0.10, 4.0).detach()

    def forward(
        self,
        expression: torch.Tensor,
        *,
        include_attention: Optional[bool] = None,
        include_source: Optional[bool] = None,
        include_edge: Optional[bool] = None,
        include_virtual: Optional[bool] = None,
        include_virtual_value: Optional[bool] = None,
        include_virtual_capacity: Optional[bool] = None,
        include_residual: Optional[bool] = None,
        metabolite_keep: Optional[torch.Tensor] = None,
        return_real: bool = True,
        return_aux: bool = False,
    ) -> Dict[str, Any]:
        attention = self.attention_enabled and (
            True if include_attention is None else bool(include_attention)
        )
        source_on = self.source_enabled and attention and (
            True if include_source is None else bool(include_source)
        )
        edge_on = self.edge_enabled and attention and (
            True if include_edge is None else bool(include_edge)
        )
        virtual_on = self.virtual_enabled and (
            True if include_virtual is None else bool(include_virtual)
        )
        virtual_value_on = virtual_on and (
            True if include_virtual_value is None else bool(include_virtual_value)
        )
        virtual_capacity_on = (
            virtual_on
            and source_on
            and self.virtual_mm_capacity_coupling > 0.0
            and (
                True
                if include_virtual_capacity is None
                else bool(include_virtual_capacity)
            )
        )
        residual_on = self.residual_enabled and (
            True if include_residual is None else bool(include_residual)
        )
        gene_capacity_source = self.encode_gene(expression, apply_dropout=False)
        gene = self.dropout(gene_capacity_source)
        real_metabolite, metabolite_raw, gm_aux = self._metabolite(
            gene,
            capacity_source=gene_capacity_source,
            source_enabled=source_on,
            edge_enabled=edge_on,
            return_aux=True,
        )
        metabolite_pre_dropout = torch.tanh(self.metabolite_opportunity(metabolite_raw))
        if metabolite_keep is None:
            keep = torch.ones(
                real_metabolite.shape[:2],
                dtype=real_metabolite.dtype,
                device=real_metabolite.device,
            )
        else:
            keep = metabolite_keep.to(real_metabolite)
            if keep.ndim == 3 and keep.shape[-1] == 1:
                keep = keep[:, :, 0]
            if keep.shape != real_metabolite.shape[:2]:
                raise ValueError("metabolite_keep has an invalid shape")
        observed = real_metabolite * keep[:, :, None]
        full_metabolite, virtual_aux = self.virtual_context(
            real_metabolite,
            keep,
            enabled=virtual_value_on,
            return_aux=True,
        )
        if isinstance(
            self.mm, NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
        ):
            real_mm_capacity_source = metabolite_pre_dropout * keep[:, :, None]
            if virtual_capacity_on:
                virtual_mm_capacity_source = self.virtual_context(
                    metabolite_pre_dropout,
                    keep,
                    enabled=True,
                    return_aux=False,
                )
                mm_capacity_source = real_mm_capacity_source + (
                    self.virtual_mm_capacity_coupling
                    * (virtual_mm_capacity_source - real_mm_capacity_source)
                )
            else:
                virtual_mm_capacity_source = real_mm_capacity_source
                mm_capacity_source = real_mm_capacity_source
        else:
            # Align the complementary branch with the knowledge representation.
            real_mm_capacity_source = observed
            virtual_mm_capacity_source = observed
            mm_capacity_source = observed
        microbe, microbe_raw, mm_aux = self._microbe(
            full_metabolite,
            capacity_source=mm_capacity_source,
            source_enabled=source_on,
            edge_enabled=edge_on,
            return_aux=True,
        )
        knowledge_logit, pool_aux = self.pool(microbe, return_aux=True)
        real_logit = None
        real_microbe = None
        if return_real:
            real_microbe = self._microbe(
                observed,
                capacity_source=real_mm_capacity_source,
                source_enabled=source_on,
                edge_enabled=edge_on,
            )
            real_logit = self.pool(real_microbe)
        residual_logit = None
        residual_aux = None
        if residual_on:
            residual_logit, residual_aux = self.residual(
                expression,
                self._configured_gm_coefficient(),
                knowledge_summary=(
                    real_microbe.mean(1) if real_microbe is not None else microbe.mean(1)
                ),
                return_aux=True,
            )
            if self.training and self.residual_variance_balance:
                self._update_residual_balance(knowledge_logit, residual_logit)
            balance = self.residual_balance_factor()
            residual_logit = balance * residual_logit
            residual_aux["variance_balance_factor"] = balance
        final_logit = (
            knowledge_logit
            if residual_logit is None
            else knowledge_logit + self.residual_fusion_weight * residual_logit
        )
        result: Dict[str, Any] = {
            "final": final_logit,
            "knowledge": knowledge_logit,
            "real": real_logit,
            "residual": residual_logit,
        }
        if return_aux:
            result["auxiliary"] = {
                "gene_state": gene,
                "gene_capacity_source_state": gene_capacity_source,
                "real_metabolite_state": real_metabolite,
                "metabolite_pre_dropout_state": metabolite_pre_dropout,
                "observed_metabolite_state": observed,
                "metabolite_capacity_source_state": mm_capacity_source,
                "real_metabolite_capacity_source_state": real_mm_capacity_source,
                "virtual_metabolite_capacity_source_state": (
                    virtual_mm_capacity_source
                ),
                "full_metabolite_state": full_metabolite,
                "microbe_state": microbe,
                "metabolite_raw": metabolite_raw,
                "microbe_raw": microbe_raw,
                "metabolite_keep": keep,
                "gm": gm_aux,
                "mm": mm_aux,
                "virtual": virtual_aux,
                "virtual_value_enabled": bool(virtual_value_on),
                "virtual_capacity_enabled": bool(virtual_capacity_on),
                "pool": pool_aux,
                "residual": residual_aux,
            }
        return result

    @torch.no_grad()
    def reset_control_rms(self) -> None:
        for control in (self.metabolite_opportunity, self.microbe_opportunity):
            control.running_mean_square.fill_(1.0)
            control.updates.zero_()
        for relation in (self.gm, self.mm):
            if isinstance(
                relation, NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
            ):
                relation.reset_source_reference()
        if self.residual_variance_balance:
            self.knowledge_logit_variance.fill_(1.0)
            self.residual_logit_variance.fill_(1.0)
            self.residual_balance_updates.zero_()

    @torch.no_grad()
    def update_control_rms(self, expression: torch.Tensor) -> None:
        if not self.training:
            raise RuntimeError("Control RMS may only be updated in training mode")
        gene = self.encode_gene(expression, apply_dropout=False)
        gm_options = {
            "source_enabled": self.source_enabled and not self.control_rms_neutral,
            "edge_enabled": self.edge_enabled and not self.control_rms_neutral,
        }
        if isinstance(
            self.gm, NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
        ):
            if not bool(self.gm.source_reference_frozen.item()):
                self.gm.update_source_reference(gene)
            gm_options["capacity_source"] = gene
        metabolite_raw = self.gm(gene, **gm_options)
        self.metabolite_opportunity.update(metabolite_raw)
        metabolite = torch.tanh(self.metabolite_opportunity(metabolite_raw))
        mm_options = {
            "source_enabled": self.source_enabled and not self.control_rms_neutral,
            "edge_enabled": self.edge_enabled and not self.control_rms_neutral,
        }
        if isinstance(
            self.mm, NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
        ):
            if not bool(self.mm.source_reference_frozen.item()):
                self.mm.update_source_reference(metabolite)
            mm_options["capacity_source"] = metabolite
        microbe_raw = self.mm(metabolite, **mm_options)
        self.microbe_opportunity.update(microbe_raw)

    @torch.no_grad()
    def freeze_source_references(self) -> None:
        """Freeze fold-training capacity statistics; a no-op for older relations."""
        for relation in (self.gm, self.mm):
            if isinstance(
                relation, NodewiseDynamicMultiHeadSignedBudgetAttentionRelation
            ):
                relation.freeze_source_reference()

    def regularization(self) -> Mapping[str, torch.Tensor]:
        return {
            "attention": self.gm.regularization() + self.mm.regularization(),
            "virtual": self.virtual_context.regularization(),
        }
