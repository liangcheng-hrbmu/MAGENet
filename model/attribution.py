"""Gradient and perturbation attribution with consensus node rankings."""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import textwrap
import types
import numpy as np
import torch
from torch.nn import functional as F
import pandas as pd
from scipy.stats import rankdata


def _replay_method(instance, old, new):
    original = instance.forward
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(old) != 1:
        raise RuntimeError('Validated stop-gradient expression no longer matches source')
    namespace = dict(original.__func__.__globals__)
    exec(compile(source.replace(old, new), '<forward-identical-attribution>', 'exec'), namespace)
    return types.MethodType(namespace['forward'], instance)


@contextmanager
def full_forward_derivative(model):
    """Interpret the actual prediction function, including training-detached paths."""
    modules = [model.residual, model.virtual_context]
    existed = [('forward' in m.__dict__, m.__dict__.get('forward')) for m in modules]
    methods = [
        _replay_method(model.residual, 'knowledge_summary.detach()', 'knowledge_summary'),
        _replay_method(model.virtual_context,
            'reference_norm / innovation_norm.clamp_min(1e-8),\n            ).detach()',
            'reference_norm / innovation_norm.clamp_min(1e-8),\n            )')]
    try:
        for m, method in zip(modules, methods):
            m.forward = method
        yield
    finally:
        for m, (had_override, override) in zip(modules, existed):
            if had_override:
                m.forward = override
            else:
                delattr(m, 'forward')


def gene_probability_sensitivity(model, x, batch=128):
    if model.training or x.dtype != torch.float64 or next(model.parameters()).dtype != torch.float64:
        raise ValueError('Interpretation requires eval mode and FP64 model/input')
    if model.normalize_encoder_scale:
        raise ValueError('Chain-rule mapping is locked to the unnormalized tanh encoder')
    signed, logits, error, chain_error = [], [], 0., 0.
    with torch.no_grad():
        native = torch.cat([model(x[i:i+batch])['final'].reshape(-1) for i in range(0, len(x), batch)])
    with full_forward_derivative(model):
        for start in range(0, len(x), batch):
            bx = x[start:start+batch]
            out = model(bx, return_aux=True)
            z = out['final'].reshape(-1)
            h = out['auxiliary']['gene_capacity_source_state']
            dh = torch.autograd.grad(z.sum(), h, retain_graph=True)[0]
            dz = dh * (1-h.square()) * bx.clamp(-5,5)[:,:,None]
            p = z.sigmoid()
            dp = dz * (p*(1-p))[:,None,None]
            # Independent aggregate chain-rule check against autograd parameters.
            direct = torch.autograd.grad(p.sum(), model.encoder.scale)[0]
            chain_error = max(chain_error, float((dp.sum(0)-direct).detach().abs().max()))
            assert torch.allclose(dp.sum(0), direct, atol=1e-11, rtol=1e-9)
            signed.append(dp.sum(2).detach().cpu().numpy())
            logits.append(z.detach().cpu().numpy())
            error = max(error, float((z.detach()-native[start:start+len(bx)]).abs().max()))
    assert error == 0.0
    derivative = np.concatenate(signed)
    return np.abs(derivative).mean(0), derivative, np.concatenate(logits), dict(
        forward_max_abs_difference=error, chain_rule_max_abs_error=chain_error)


@torch.no_grad()
def finite_difference_gene(model, x, index, epsilon=1e-5, batch=128):
    target = model.encoder.scale[index]
    original = target.clone()
    pieces = []
    try:
        for start in range(0,len(x),batch):
            bx = x[start:start+batch]
            target.copy_(original+epsilon)
            plus = model(bx)['final'].reshape(-1).sigmoid()
            target.copy_(original-epsilon)
            minus = model(bx)['final'].reshape(-1).sigmoid()
            pieces.append(((plus-minus)/(2*epsilon)).cpu().numpy())
    finally:
        target.copy_(original)
    return np.concatenate(pieces)


EPSILON = 1e-5


BATCH = 256


def parameters(model):
    return dict(gene=model.encoder.scale, metabolite=model.gm.edge_logit, microbe=model.mm.edge_logit)


def gradient(model, x, y, batch=BATCH):
    model.eval()
    params = parameters(model)
    original_flags = {name:p.requires_grad for name,p in model.named_parameters()}
    selected = {id(p) for p in params.values()}
    try:
        for p in model.parameters():
            p.requires_grad_(id(p) in selected)
        totals = {name:torch.zeros_like(p, dtype=torch.float64) for name,p in params.items()}
        for start in range(0, len(x), batch):
            output = model(x[start:start+batch])['final'].reshape(-1)
            loss = F.binary_cross_entropy_with_logits(output, y[start:start+batch], reduction='sum') / len(x)
            pieces = torch.autograd.grad(loss, tuple(params.values()))
            for name, piece in zip(params, pieces):
                totals[name].add_(piece.detach().double())
        return totals
    finally:
        for name,p in model.named_parameters():
            p.requires_grad_(original_flags[name])


def reduce_gradient(model, grads):
    return dict(gene=grads['gene'].abs().mean(1).cpu().numpy(),
                metabolite=(grads['metabolite']*model.gm.mask[None]).abs().mean((0,1)).cpu().numpy(),
                microbe=(grads['microbe']*model.mm.mask[None]).abs().mean((0,1)).cpu().numpy())


def column(model, layer, node):
    p = parameters(model)[layer]
    if layer == 'gene':
        return p[node, :], torch.ones_like(p[node, :])
    relation = model.gm if layer == 'metabolite' else model.mm
    return p[:, :, node], relation.mask[:, node][None].expand_as(p[:, :, node])


@torch.no_grad()
def perturbation(model, x, layer, node, batch=BATCH):
    target, mask = column(model, layer, node)
    original = target.clone()
    differences = []
    try:
        for start in range(0, len(x), batch):
            bx = x[start:start+batch]
            target.copy_(original + EPSILON*mask)
            plus = model(bx)['final'].sigmoid().reshape(-1)
            target.copy_(original - EPSILON*mask)
            minus = model(bx)['final'].sigmoid().reshape(-1)
            differences.append((plus-minus).abs().double())
    finally:
        target.copy_(original)
    diff = torch.cat(differences)
    return float(diff.mean()/(2*EPSILON)), float((diff == 0).double().mean())


class GeneStateGate:
    def __init__(self, model):
        p = next(model.parameters())
        self.alpha = torch.ones(model.nodes["genes"], device=p.device, dtype=p.dtype, requires_grad=True)
        self.last_base = None
        self.handle = model.encoder.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        self.last_base = output
        return output * self.alpha[None, :, None]

    @torch.no_grad()
    def set(self, index=None, value=1.0):
        self.alpha.fill_(1.0)
        if index is not None:
            self.alpha[index] = value

    def close(self):
        self.handle.remove()


def consensus_table(names, gradient_scores, perturbation_scores):
    gradient_scores, perturbation_scores = map(np.asarray, (gradient_scores, perturbation_scores))
    if gradient_scores.shape != (len(names),) or perturbation_scores.shape != (len(names),):
        raise ValueError("Each node requires two scalar importance scores")
    if not np.isfinite(gradient_scores).all() or not np.isfinite(perturbation_scores).all():
        raise ValueError("Importance scores must be finite")
    a, b = rankdata(-gradient_scores, method="average"), rankdata(-perturbation_scores, method="average")
    return pd.DataFrame(dict(node=names, gradient_score=gradient_scores, perturbation_score=perturbation_scores,
                             gradient_rank=a, perturbation_rank=b, mean_rank=(a + b) / 2)).sort_values(
                                 ["mean_rank", "node"], kind="stable").reset_index(drop=True)


def gene_state_ranking(model, x, batch_size=128, attenuation=0.2):
    """Rank genes by |dP/dalpha| and |P(alpha=1)-P(alpha=.8)|/.2.

    Applies the state multiplier to the gene encoder output. The expression
    residual retains its original input. Inputs/model must use float64.
    """
    if model.training or next(model.parameters()).dtype != torch.float64 or x.dtype != torch.float64:
        raise ValueError("Use model.double().eval() and FP64 standardized inputs")
    if model.normalize_encoder_scale or not 0 < attenuation < 1:
        raise ValueError("Requires the unnormalized encoder and attenuation in (0,1)")
    gate = GeneStateGate(model)
    try:
        derivatives, baseline = [], []
        with full_forward_derivative(model):
            for start in range(0, len(x), batch_size):
                out = model(x[start:start + batch_size], return_aux=True)
                p = out["final"].sigmoid()
                h = out["auxiliary"]["gene_capacity_source_state"]
                grad = torch.autograd.grad(p.sum(), h)[0]
                derivatives.append((grad * gate.last_base).sum(2).detach().cpu().numpy())
                baseline.append(p.detach().cpu().numpy())
        base = np.concatenate(baseline)
        finite = []
        with torch.no_grad():
            for i in range(model.nodes["genes"]):
                gate.set(i, 1 - attenuation)
                changed = np.concatenate([model(x[j:j + batch_size])["final"].sigmoid().cpu().numpy()
                                          for j in range(0, len(x), batch_size)])
                finite.append(np.abs(base - changed).mean() / attenuation)
        gradient = np.abs(np.concatenate(derivatives)).mean(0)
        return consensus_table(model.graph.genes, gradient, finite)
    finally:
        gate.close()
