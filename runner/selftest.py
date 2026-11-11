"""Sanity checks. Run from autolab:  python -m runner.selftest
The numpy parts need only numpy; the torch parts run when torch is installed."""
import numpy as np
from .tasks import make_recall_batch, KEY0, VAL0, QUERY, PAD
from .quant import quantize_np, torch


def test_tasks():
    rng = np.random.default_rng(0)
    for d in (64, 256, 1024):
        inp, tgt = make_recall_batch(rng, 5, d)
        assert inp.shape == tgt.shape == (5, 8 + d + 12 - 1)
        for i in range(5):
            bind = {inp[i, 2 * j] - KEY0: inp[i, 2 * j + 1] - VAL0 for j in range(4)}
            pos = np.where(tgt[i] != -100)[0]
            assert len(pos) == 4
            for p in pos:                       # input at p is the queried key, target is its value
                assert inp[i, p - 1] == QUERY
                assert bind[inp[i, p] - KEY0] == tgt[i, p] - VAL0
        assert (inp[:, 8:8 + d] < 256).all()    # filler is plain bytes
    print("tasks ok")


def test_quant_np():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(3, 8, 4)).astype(np.float32)
    for bits in (8, 4, 2):
        q = quantize_np(h, bits)
        for i in range(3):
            assert len(np.unique(q[i])) <= 2 ** bits
            lo, hi = min(h[i].min(), 0), max(h[i].max(), 0)
            step = (hi - lo) / (2 ** bits - 1)
            assert np.abs(q[i] - h[i]).max() <= step / 2 + 1e-6
    assert (quantize_np(np.zeros((1, 4, 2), np.float32), 4) == 0).all()          # zeros stay exact
    keep = np.array([True, False, False, True, False, False, False, False])
    q = quantize_np(h, 2, keep)
    assert (q[:, keep] == h[:, keep]).all()                                       # kept channels untouched
    assert len(np.unique(q[0][~keep])) <= 4
    assert (quantize_np(h, 4, np.ones(8, bool)) == h).all()
    # more bits never hurts (error is monotone here)
    errs = [np.abs(quantize_np(h, b) - h).mean() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]
    print("quant (numpy) ok")


def test_torch():
    if torch is None:
        print("torch not installed: SKIPPED model/quant-parity tests (run these on the GPU machine)")
        return
    from .quant import fake_quant
    from .model import SSMLM
    from .evaluate import StateHook
    from .tasks import VOCAB
    rng = np.random.default_rng(2)
    h = rng.normal(size=(3, 8, 4)).astype(np.float32)
    keep = np.array([True, False, False, True, False, False, False, False])
    for bits in (8, 4, 2):
        for kp in (None, keep):
            ref = quantize_np(h, bits, kp)
            got = fake_quant(torch.from_numpy(h), bits, None if kp is None else torch.from_numpy(kp)).numpy()
            assert np.allclose(ref, got, atol=1e-6), (bits, kp is None)
    ref = quantize_np(h, 4, static=(-2.0, 3.0))
    got = fake_quant(torch.from_numpy(h), 4, static=(-2.0, 3.0)).numpy()
    assert np.allclose(ref, got, atol=1e-6)
    torch.manual_seed(0)
    m = SSMLM(32, 2, 2, 8).eval()
    x = torch.randint(0, VOCAB, (2, 40))
    base = m(x)
    assert base.shape == (2, 40, VOCAB)
    ident = m(x, [StateHook(), StateHook()])
    assert torch.allclose(base, ident, atol=1e-6)                                # no-op hook changes nothing
    q8 = m(x, [StateHook(bits=8), StateHook(bits=8)])
    q2 = m(x, [StateHook(bits=2), StateHook(bits=2)])
    assert (q8 - base).abs().mean() < (q2 - base).abs().mean()                   # fewer bits, larger deviation
    ab = torch.ones(2, m.d_inner, 1); ab[:, 0, 0] = 0
    assert not torch.allclose(m(x, [StateHook(ablate=ab), None]), base)          # ablation has an effect
    # causality: changing a late token must not change earlier logits
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % VOCAB
    assert torch.allclose(m(x2)[:, :-1], base[:, :-1], atol=1e-5)
    print("torch model/quant ok,", m.n_params(), "params")


if __name__ == "__main__":
    test_tasks(); test_quant_np(); test_torch()
