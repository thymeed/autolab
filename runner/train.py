"""Train the full-precision model from scratch (one model per seed)."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .tasks import make_recall_batch, lm_batch

TRAIN_DISTANCES = (16, 64, 256, 1024)


def train(model, text, seed, steps, batch, lm_len, lr, device, log=print):
    rng = np.random.default_rng(10_000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    warm = max(1, steps // 30)
    model.train()
    for step in range(steps):
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * max(f, 0.05)
        if rng.random() < 0.5:
            d = int(rng.choice(TRAIN_DISTANCES))
            x, y = make_recall_batch(rng, batch, d)
            kind = f"recall{d}"
        else:
            x, y = lm_batch(rng, text, batch, lm_len)
            kind = "lm"
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log(f"  seed {seed} step {step}/{steps} {kind} loss {loss.item():.4f}")
    model.eval()
    return model
