from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from .network import MAGENet

def build_model(
    graph: Any,
    config: Mapping[str, Any],
    flags: Mapping[str, bool],
    gm_basis: np.ndarray,
    mm_basis: np.ndarray,
) -> MAGENet:
    return MAGENet(
        graph,
        gm_basis,
        mm_basis,
        width=int(config["width"]),
        attention_enabled=bool(flags["attention_enabled"]),
        source_enabled=bool(flags["source_enabled"]),
        edge_enabled=bool(flags["edge_enabled"]),
        virtual_enabled=bool(flags["virtual_enabled"]),
        residual_enabled=bool(flags["residual_enabled"]),
        dropout=float(config["dropout"]),
        residual_dropout=float(config["residual_dropout"]),
        gm_maximum_edge_ratio=float(config["gm_maximum_edge_ratio"]),
        mm_maximum_edge_ratio=float(config["mm_maximum_edge_ratio"]),
        maximum_source_ratio=float(config["maximum_source_ratio"]),
        minimum_opportunity_attenuation=float(config["minimum_opportunity_attenuation"]),
        peer_top_k=int(config["peer_top_k"]),
        maximum_virtual_scale=float(config["maximum_virtual_scale"]),
        virtual_scale_init=min(
            float(config["virtual_scale_init"]),
            0.5 * float(config["maximum_virtual_scale"]),
        ),
        maximum_residual_scale=float(config["maximum_residual_scale"]),
        residual_scale_init=float(config["residual_scale_init"]),
        residual_mode=str(config["residual_mode"]),
        relation_kind=str(config.get("relation_kind", "normalized")),
        relation_heads=int(config.get("relation_heads", 3)),
        maximum_head_gate_ratio=float(
            config.get("maximum_head_gate_ratio", 4.0)
        ),
        maximum_dynamic_source_ratio=float(
            config.get("maximum_dynamic_source_ratio", 4.0)
        ),
        maximum_fixed_source_ratio=float(
            config.get("maximum_fixed_source_ratio", 1.5)
        ),
        dynamic_source_context_mode=str(
            config.get("dynamic_source_context_mode", "local_node")
        ),
        target_opportunity_power=float(config.get("target_opportunity_power", 0.0)),
        gm_target_opportunity_power=(
            None
            if config.get("gm_target_opportunity_power") is None
            else float(config["gm_target_opportunity_power"])
        ),
        mm_target_opportunity_power=(
            None
            if config.get("mm_target_opportunity_power") is None
            else float(config["mm_target_opportunity_power"])
        ),
        virtual_clean_mode=str(config.get("virtual_clean_mode", "multiplicative")),
        virtual_targetwise_scale=bool(config.get("virtual_targetwise_scale", False)),
        virtual_samplewise_gate=bool(config.get("virtual_samplewise_gate", False)),
        virtual_samplewise_gate_mode=str(
            config.get("virtual_samplewise_gate_mode", "unbounded_logit")
        ),
        virtual_maximum_clean_gate_ratio=float(
            config.get("virtual_maximum_clean_gate_ratio", 4.0)
        ),
        virtual_learnable_peers=bool(config.get("virtual_learnable_peers", False)),
        virtual_maximum_peer_ratio=float(config.get("virtual_maximum_peer_ratio", 4.0)),
        virtual_mm_capacity_coupling=float(
            config.get("virtual_mm_capacity_coupling", 0.0)
        ),
        virtual_samplewise_peer_attention=bool(
            config.get("virtual_samplewise_peer_attention", False)
        ),
        virtual_peer_attention_strength_init=float(
            config.get("virtual_peer_attention_strength_init", 0.25)
        ),
        virtual_peer_attention_query_mode=str(
            config.get("virtual_peer_attention_query_mode", "target")
        ),
        normalize_encoder_scale=bool(config.get("normalize_encoder_scale", False)),
        control_rms_neutral=bool(config.get("control_rms_neutral", False)),
        residual_fusion_weight=float(config.get("residual_fusion_weight", 1.0)),
        pool_kind=str(config.get("pool_kind", "direct")),
        residual_variance_balance=bool(
            config.get("residual_variance_balance", False)
        ),
        shuffle_virtual_peers=bool(flags["shuffle_virtual_peers"]),
        peer_seed=int(config.get('peer_seed', 1992)),
    )
