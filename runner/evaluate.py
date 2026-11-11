"""Hooks on the recurrent state, evaluation, static calibration, channel ranking."""
import numpy as np
import torch
import torch.nn.functional as F
from .quant import fake_quant
from .tasks import VAL0, NVALS, make_recall_batch, lm_batch, DISTANCES


class StateHook:
    """Applied to h (B, C, N) after every recurrent update, in this order:
    ablate (multiply by (B,C,1) 0/1 mask) -> record range -> quantize."""

    def __init__(self, bits=None, keep=None, static=None, ablate=None, record=None):
        self.bits, self.keep, self.static = bits, keep, static
        self.ablate, self.record = ablate, record

    def __call__(self, h):
        if self.ablate is not None:
            h = h * self.ablate
        if self.record is not None:
            self.record["lo"] = min(self.record["lo"], float(h.min()))
            self.record["hi"] = max(self.record["hi"], float(h.max()))
        if self.bits is not None:
            h = fake_quant(h, self.bits, self.keep, self.static)
        return h


def make_recall_set(rng, n, distances=DISTANCES):
    return {d: tuple(torch.from_numpy(a) for a in make_recall_batch(rng, n, d)) for d in distances}


def make_lm_set(rng, data, n_batches, batch, length):
    return [tuple(torch.from_numpy(a) for a in lm_batch(rng, data, batch, length))
            for _ in range(n_batches)]


def recall_correct(logits, tgt):
    """Per-sequence fraction of query values recalled exactly (argmax over the value tokens)."""
    mask = tgt != -100
    pred = logits[..., VAL0:VAL0 + NVALS].argmax(-1) + VAL0
    return ((pred == tgt) & mask).sum(1).float() / mask.sum(1).clamp_min(1)


@torch.no_grad()
def eval_recall(model, recall_set, hooks, device, chunk=64):
    out = {}
    for d, (inp, tgt) in recall_set.items():
        accs = []
        for i in range(0, inp.shape[0], chunk):
            x, y = inp[i:i + chunk].to(device), tgt[i:i + chunk].to(device)
            accs.append(recall_correct(model(x, hooks), y).cpu())
        out[d] = float(torch.cat(accs).mean())
    return out


@torch.no_grad()
def eval_lm(model, lm_set, hooks, device):
    """Mean cross-entropy in nats per byte."""
    tot = 0.0
    for x, y in lm_set:
        logits = model(x.to(device), hooks)
        tot += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.to(device).reshape(-1)))
    return tot / len(lm_set)


@torch.no_grad()
def calibrate_static(model, recall_set, lm_set, device):
    """Fixed per-layer (lo, hi) of the full-precision state on the calibration data."""
    recs = [{"lo": float("inf"), "hi": float("-inf")} for _ in model.blocks]
    hooks = [StateHook(record=r) for r in recs]
    eval_recall(model, recall_set, hooks, device)
    eval_lm(model, lm_set, hooks, device)
    return [(r["lo"], r["hi"]) for r in recs]


@torch.no_grad()
def rank_channels(model, calib_set, device, group=16):
    """Importance[l, c] = baseline recall - recall with channel c of layer l zeroed after
    every update, on the calibration set. Variants are batched along the batch dimension."""
    L, C = len(model.blocks), model.d_inner
    base = np.mean(list(eval_recall(model, calib_set, None, device).values()))
    scores = torch.zeros(L, C)
    for l in range(L):
        for c0 in range(0, C, group):
            cs = list(range(c0, min(C, c0 + group)))
            K = len(cs)
            acc = torch.zeros(K)
            for d, (inp, tgt) in calib_set.items():
                n = inp.shape[0]
                x, y = inp.repeat(K, 1).to(device), tgt.repeat(K, 1).to(device)
                ab = torch.ones(K * n, C, 1, device=device)
                for j, c in enumerate(cs):
                    ab[j * n:(j + 1) * n, c, 0] = 0
                hooks = [None] * L
                hooks[l] = StateHook(ablate=ab)
                per_seq = recall_correct(model(x, hooks), y).cpu().view(K, n)
                acc += per_seq.mean(1) / len(calib_set)
            scores[l, c0:c0 + K] = float(base) - acc
    return scores


def keep_masks_from_scores(scores, frac, device):
    """Global top-`frac` of (layer, channel) pairs by importance -> list of bool (C,) per layer."""
    L, C = scores.shape
    k = int(round(frac * L * C))
    flat = torch.zeros(L * C, dtype=torch.bool)
    if k > 0:
        flat[torch.topk(scores.flatten(), k).indices] = True
    return [m.to(device) for m in flat.view(L, C)]


def build_hooks(n_layers, bits, keep_masks, static_ranges, scale_mode):
    hooks = []
    for l in range(n_layers):
        keep = None if keep_masks is None else keep_masks[l]
        static = static_ranges[l] if scale_mode == "static" else None
        hooks.append(StateHook(bits=bits, keep=keep, static=static))
    return hooks
