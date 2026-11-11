"""Post-training uniform asymmetric min-max quantization of the recurrent state.

Round-to-nearest (ties to even), dynamic scales by default. "Per-tensor" means one
(lo, hi) per sequence over all quantized (channel, state) entries, so examples in a
batch never influence each other. The range always includes 0 so the zero point is
an integer in [0, 2^bits - 1] and exact zeros stay exact. `keep` marks channels left
in full precision; they are excluded from the min/max and passed through untouched.
"""
import numpy as np

try:
    import torch
except ImportError:  # numpy reference still usable (selftest)
    torch = None


def quantize_np(h, bits, keep=None, static=None):
    """Reference implementation. h: (B, C, N) float array. keep: bool (C,) or None."""
    levels = 2 ** bits - 1
    h = np.asarray(h, dtype=np.float32)
    if keep is not None and keep.all():
        return h.copy()
    if static is None:
        src = h if keep is None else h[:, ~keep, :]
        lo = src.min(axis=(1, 2), keepdims=True)
        hi = src.max(axis=(1, 2), keepdims=True)
    else:
        lo, hi = (np.float32(static[0]), np.float32(static[1]))
    lo, hi = np.minimum(lo, 0), np.maximum(hi, 0)
    scale = np.maximum(hi - lo, 1e-8) / levels
    zp = np.clip(np.round(-lo / scale), 0, levels)
    q = np.clip(np.round(h / scale) + zp, 0, levels)
    deq = (q - zp) * scale
    if keep is None:
        return deq.astype(np.float32)
    return np.where(keep[None, :, None], h, deq).astype(np.float32)


def fake_quant(h, bits, keep=None, static=None):
    """Torch version of quantize_np. h: (B, C, N); keep: bool tensor (C,); static: (lo, hi)."""
    levels = 2 ** bits - 1
    if keep is not None and bool(keep.all()):
        return h
    if static is None:
        src = h if keep is None else h[:, ~keep, :]
        lo = src.amin(dim=(1, 2), keepdim=True)
        hi = src.amax(dim=(1, 2), keepdim=True)
    else:
        lo = torch.as_tensor(static[0], dtype=h.dtype, device=h.device).view(1, 1, 1)
        hi = torch.as_tensor(static[1], dtype=h.dtype, device=h.device).view(1, 1, 1)
    lo, hi = torch.clamp(lo, max=0.0), torch.clamp(hi, min=0.0)
    scale = (hi - lo).clamp_min(1e-8) / levels
    zp = torch.round(-lo / scale).clamp(0, levels)
    q = torch.clamp(torch.round(h / scale) + zp, 0, levels)
    deq = (q - zp) * scale
    if keep is None:
        return deq
    return torch.where(keep.view(1, -1, 1), h, deq)
