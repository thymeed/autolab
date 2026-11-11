"""Run the quantization experiment: train FP model per seed, evaluate baseline and every
quantization arm on identical data, write runs/<name>/seed<k>.json.

  python -m runner.run --name smoke --seeds 0 --train-steps 50 --d-model 64 --layers 2 \
      --bits 8 4 2 --keep 0 0.15 --eval-n 16 --calib-n 4 --lm-eval-batches 2
  python -m runner.run --name round1 --tinystories C:\\path\\TinyStories.txt
Run from the autolab folder.
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from .tasks import load_text, DISTANCES
from .model import SSMLM
from .train import train
from . import evaluate as E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tinystories", default=None, help="plain-text TinyStories file; omit only for smoke tests")
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lm-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    p.add_argument("--keep", type=float, nargs="+", default=[0, 0.15, 0.2])
    p.add_argument("--scales", nargs="+", default=["dynamic", "static"], choices=["dynamic", "static"])
    p.add_argument("--eval-n", type=int, default=256, help="recall sequences per distance")
    p.add_argument("--calib-n", type=int, default=16, help="recall sequences per distance for ranking/static scales")
    p.add_argument("--lm-eval-batches", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    out = os.path.join("runs", a.name)
    os.makedirs(out, exist_ok=True)
    dev = a.device
    for seed in a.seeds:
        t0 = time.time()
        torch.manual_seed(seed)
        rng_text = np.random.default_rng(seed)
        tr, ca, ev, source = load_text(a.tinystories, rng_text)
        if source != "tinystories":
            print("WARNING: no TinyStories file given -> synthetic fallback text, results are SMOKE ONLY")
        model = SSMLM(a.d_model, a.layers, 2, a.d_state).to(dev)
        n_params = model.n_params()
        assert n_params < 20_000_000, n_params
        print(f"seed {seed}: {n_params/1e6:.2f}M params on {dev}")
        train(model, tr, seed, a.train_steps, a.batch, a.lm_len, a.lr, dev)

        # held-out data: eval and calibration are disjoint (different text split + rng streams)
        r_eval = np.random.default_rng(20_000 + seed)
        r_cal = np.random.default_rng(30_000 + seed)
        eval_recall = E.make_recall_set(r_eval, a.eval_n)
        eval_lm = E.make_lm_set(r_eval, ev, a.lm_eval_batches, a.batch, a.lm_len)
        cal_recall = E.make_recall_set(r_cal, a.calib_n)
        cal_lm = E.make_lm_set(r_cal, ca, max(2, a.lm_eval_batches // 4), a.batch, a.lm_len)

        def score(hooks):
            acc = E.eval_recall(model, eval_recall, hooks, dev)
            return {"acc": {str(d): v for d, v in acc.items()},
                    "acc_avg": float(np.mean(list(acc.values()))),
                    "loss": E.eval_lm(model, eval_lm, hooks, dev)}

        base = score(None)
        print(f"  baseline acc {base['acc_avg']:.3f} loss {base['loss']:.4f} (chance acc {1/64:.3f})")
        static = E.calibrate_static(model, cal_recall, cal_lm, dev)
        scores = E.rank_channels(model, cal_recall, dev)
        results = []
        for bits in a.bits:
            for frac in a.keep:
                masks = E.keep_masks_from_scores(scores, frac, dev) if frac > 0 else None
                for sm in a.scales:
                    r = score(E.build_hooks(len(model.blocks), bits, masks, static, sm))
                    r.update(bits=bits, keep=frac, scale=sm)
                    results.append(r)
                    print(f"  bits {bits} keep {frac} {sm}: acc {r['acc_avg']:.3f} loss {r['loss']:.4f}")
        meta = dict(vars(a), seed=seed, text_source=source, n_params=n_params,
                    torch=torch.__version__, distances=list(DISTANCES),
                    seconds=round(time.time() - t0))
        with open(os.path.join(out, f"seed{seed}.json"), "w") as f:
            json.dump(dict(meta=meta, baseline=base, importance=scores.tolist(),
                           static_ranges=static, results=results), f)
        print(f"  wrote seed{seed}.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
