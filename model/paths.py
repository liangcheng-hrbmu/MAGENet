"""Knowledge-network path messages and path contribution scores."""
from __future__ import annotations

import numpy as np
import torch
import pandas as pd


def tanh_rescale(raw, attenuation):
    """tanh(a*z)/z with its continuous a limit at zero, never tanh(z)/z."""
    att = attenuation[None, :, None]
    safe = torch.where(raw != 0, raw, torch.ones_like(raw))
    return torch.where(raw != 0, torch.tanh(raw * att) / safe, att.expand_as(raw))


@torch.no_grad()
def collect_states(model, x, device, batch_size):
    """Store only sufficient observed states; native branches are also audited."""
    names = ("vg", "cg", "cm", "rm", "rb", "um", "ub", "real_logit", "final_logit", "knowledge_logit")
    lists = {key: [] for key in names}
    replay_max = 0.0
    native_real_max = 0.0
    m_att = model.metabolite_opportunity.attenuation()
    b_att = model.microbe_opportunity.attenuation()
    ag = model.gm.effective_head_signed_allocation(True)
    am = model.mm.effective_head_signed_allocation(True)
    for start in range(0, len(x), batch_size):
        bx = torch.as_tensor(x[start:start + batch_size], dtype=torch.float64, device=device)
        native = model(bx, return_aux=False)
        output = model(bx, include_virtual=False, include_residual=False, return_aux=True)
        aux = output["auxiliary"]
        vg = model.gm.value(aux["gene_state"])
        cg = aux["gm"]["source_capacity"]
        cm = aux["mm"]["source_capacity"]
        ug = torch.einsum("bshd,bhs,hst->bthd", vg.reshape(len(bx), len(model.gm.mask), model.gm.heads, model.gm.head_width), cg, ag).reshape_as(aux["metabolite_raw"])
        rm = tanh_rescale(ug, m_att)
        met = ug * rm
        vm = model.mm.value(met).reshape(len(bx), len(model.mm.mask), model.mm.heads, model.mm.head_width)
        ub = torch.einsum("bshd,bhs,hst->bthd", vm, cm, am).reshape_as(aux["microbe_raw"])
        rb = tanh_rescale(ub, b_att)
        replay = model.pool(ub * rb)
        replay_max = max(replay_max, float((replay - output["real"]).abs().max()))
        native_real_max = max(native_real_max, float((native["real"] - output["real"]).abs().max()))
        arrays = dict(vg=vg, cg=cg, cm=cm, rm=rm, rb=rb, um=ug, ub=ub, real_logit=output["real"], final_logit=native["final"], knowledge_logit=native["knowledge"])
        for name, value in arrays.items():
            lists[name].append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in lists.items()}, dict(
        manual_real_replay_max_abs_error=replay_max,
        native_real_vs_virtual_and_residual_off_max_abs_error=native_real_max,
    )


@torch.no_grad()
def scores_for_metabolite(model, states, m, gi, bi, device, batch_size):
    """Return [cell, gene-neighbor, microbe-neighbor] in fixed domain order."""
    h_g = torch.arange(model.width, device=device) // model.gm.head_width
    h_m = torch.arange(model.width, device=device) // model.mm.head_width
    ag = model.gm.effective_head_signed_allocation(True)[:, gi, m].T[:, h_g]
    am = model.mm.effective_head_signed_allocation(True)[:, m, bi].T[:, h_m]
    readout = model.pool.output.weight[0, bi, None] * model.pool.node_projection.weight[0][None, :]
    output = np.empty((len(states["vg"]), len(gi), len(bi)), dtype=np.float64)
    for start in range(0, len(output), batch_size):
        stop = min(start + batch_size, len(output))
        vg = torch.as_tensor(states["vg"][start:stop, gi], device=device)
        cg = torch.as_tensor(states["cg"][start:stop, :, gi], device=device)
        # Numpy advanced indexing retains [batch,head,source] for this slice.
        cg = cg.transpose(1, 2)[:, :, h_g]
        rm = torch.as_tensor(states["rm"][start:stop, m], device=device)
        cm = torch.as_tensor(states["cm"][start:stop, :, m], device=device)[:, h_m]
        rb = torch.as_tensor(states["rb"][start:stop, bi], device=device)
        upstream = vg * cg * ag[None] * rm[:, None, :]
        downstream = rb * readout[None] * am[None] * cm[:, None, :]
        # mm.value applies row-vector @ weight.T; reverse contraction is @weight.
        downstream_input = downstream @ model.mm.value.weight
        result = upstream @ downstream_input.transpose(1, 2)
        output[start:stop] = result.cpu().numpy()
    return output


def aggregate_paths(scores, labels, m, gi, bi, names, start_index):
    genes, metabolites, microbes = names
    a = scores[labels == 1].reshape(int((labels == 1).sum()), -1)
    b = scores[labels == 0].reshape(int((labels == 0).sum()), -1)
    c_r, c_nr = a.mean(0), b.mean(0)
    c = .5 * (c_r + c_nr)
    strength = .5 * (np.abs(a).mean(0) + np.abs(b).mean(0))
    positive = .5 * (np.maximum(a, 0).mean(0) + np.maximum(b, 0).mean(0))
    negative = .5 * (np.maximum(-a, 0).mean(0) + np.maximum(-b, 0).mean(0))
    nonzero = strength > 1e-12
    dominance = np.divide(np.maximum(positive, negative), strength, out=np.zeros_like(c), where=nonzero)
    positive_fraction = .5 * ((a > 1e-12).mean(0) + (b > 1e-12).mean(0))
    negative_fraction = .5 * ((a < -1e-12).mean(0) + (b < -1e-12).mean(0))
    sign_consistency = np.where(c >= 0, positive_fraction, negative_fraction)
    net_direction = np.where(np.abs(c) <= 1e-12, "Neutral", np.where(c > 0, "Response", "Nonresponse"))
    gene_indices = np.repeat(gi, len(bi))
    microbe_indices = np.tile(bi, len(gi))
    return pd.DataFrame(dict(
        path_id=np.arange(start_index, start_index + len(c)),
        gene_index=gene_indices, metabolite_index=m, microbe_index=microbe_indices,
        gene=np.asarray(genes)[gene_indices], metabolite_cid=metabolites[m], microbe=np.asarray(microbes)[microbe_indices],
        R=strength, C=c, C_R=c_r, C_NR=c_nr, D=c_r-c_nr,
        R_R=np.abs(a).mean(0), R_NR=np.abs(b).mean(0),
        positive_mass=positive, negative_mass=negative, dominance_fraction=dominance,
        net_mass_fraction=np.divide(np.abs(c), strength, out=np.zeros_like(c), where=nonzero),
        positive_cell_fraction_balanced=positive_fraction, negative_cell_fraction_balanced=negative_fraction,
        sign_consistency_balanced=sign_consistency, net_direction=net_direction,
        positive_cell_fraction_R=(a > 1e-12).mean(0), positive_cell_fraction_NR=(b > 1e-12).mean(0),
        C_sign_concordant_across_classes=np.sign(c_r)==np.sign(c_nr),
        class_dependent_direction=(c_r > 1e-12)&(c_nr < -1e-12) | (c_r < -1e-12)&(c_nr > 1e-12),
    ))


@torch.no_grad()
def attention_support(model, paths):
    """Positive head-pair support; path signs come from class-mean messages."""
    ng = model.gm.node_gate(True).cpu().numpy()
    nm = model.mm.node_gate(True).cpu().numpy()
    eg = model.gm.edge_gate(True).cpu().numpy()
    em = model.mm.edge_gate(True).cpu().numpy()
    gg, gm = ng[None, :, None] * eg, nm[None, :, None] * em
    hin, hout = model.gm.heads, model.mm.heads
    w = model.mm.value.weight.detach().cpu().numpy()
    energy = np.square(w.reshape(hout, model.mm.head_width, hin, model.gm.head_width)).sum(axis=(1, 3))
    if not np.isfinite(energy).all() or energy.sum() <= 0:
        raise ValueError("MM cross-head energy must be finite and nonzero")
    omega = energy / energy.sum()
    maskg, maskm = model.graph.gm.astype(bool), model.graph.mm.astype(bool)
    q = np.empty(len(paths), dtype=np.float64)
    for m, positions in paths.groupby("metabolite_index", sort=True).indices.items():
        block = paths.iloc[positions]
        gi, bi = block.gene_index.to_numpy(), block.microbe_index.to_numpy()
        if not maskg[gi, m].all() or not maskm[m, bi].all():
            raise ValueError("Path outside prior support")
        # Gather head weights for each supported path.
        q[positions] = np.einsum("ph,kh,kp->p", gg[:, gi, m].T, omega, gm[:, m, bi])
    if not np.isfinite(q).all() or not (q > 0).all():
        raise ValueError("Invalid attention support")
    return q


def add_pcs(model, paths):
    paths = paths.copy().reset_index(drop=True)
    paths["D"] = paths.C_R - paths.C_NR
    paths["Q"] = attention_support(model, paths)
    paths["AttentionMultiplier"] = paths.Q ** .25
    paths["PCS"] = paths.D * paths.AttentionMultiplier
    paths["AbsolutePCS"] = paths.PCS.abs()
    paths["Association"] = np.where(paths.D > 1e-12, "Responder-associated",
                                     np.where(paths.D < -1e-12, "Non-responder-associated", "Neutral"))
    paths = paths.sort_values(["AbsolutePCS", "R", "path_id"], ascending=[False, False, True], kind="stable")
    paths["PCS_rank"] = np.arange(1, len(paths) + 1)
    return paths.reset_index(drop=True)


def compute_pcs(model, x, labels, batch_size=128):
    """Compute all supported paths; leave virtual/residual out of path messages.

    Original per-cell capacities and activation rescaling are retained. Complete
    unweighted messages reconstruct the real-branch logit minus readout biases;
    PCS itself is a ranking score, not an additive decomposition of final logits.
    """
    if model.training or next(model.parameters()).dtype != torch.float64:
        raise ValueError("Use model.double().eval() for path analysis")
    labels = np.asarray(labels)
    if labels.shape != (len(x),) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Supply both classes, encoded non-responder=0 and responder=1")
    if model.normalize_encoder_scale or model.pool_kind != "direct" or model.virtual_mm_capacity_coupling != 0:
        raise ValueError("Path decomposition requires the default branch configuration")
    device = next(model.parameters()).device
    states, qa = collect_states(model, x, device, batch_size)
    frames, offset = [], 0
    total = np.zeros(len(x), dtype=np.float64)
    names = model.graph.genes, model.graph.metabolites, model.graph.microbes
    for m in range(model.nodes["metabolites"]):
        gi = np.flatnonzero(model.graph.gm[:, m])
        bi = np.flatnonzero(model.graph.mm[m])
        scores = scores_for_metabolite(model, states, m, gi, bi, device, batch_size)
        total += scores.sum((1, 2))
        frame = aggregate_paths(scores, labels, m, gi, bi, names, offset)
        frames.append(frame)
        offset += len(frame)
    with torch.no_grad():
        bias = float((model.pool.output.bias + model.pool.node_projection.bias * model.pool.output.weight.sum()).item())
    qa["path_sum_max_abs_error"] = float(np.max(np.abs(total + bias - states["real_logit"])))
    if qa["path_sum_max_abs_error"] > 1e-8:
        raise RuntimeError("Path messages did not reconstruct the real-branch output")
    paths = add_pcs(model, pd.concat(frames, ignore_index=True))
    return paths, qa
