"""Summarise runs/<name>/seed*.json against FINDING.md's rules.

  python -m runner.aggregate runs/round1
Prints and writes summary.md. Anything failing a gate is labelled PRELIMINARY
OBSERVATION, never a finding.
"""
import glob
import json
import os
import sys
import numpy as np

ACC_COLLAPSE = 0.10    # 10 percentage points
LOSS_COLLAPSE = 0.05   # nats (per byte here)


def main(path):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(path, "seed*.json")))]
    if not runs:
        sys.exit(f"no seed*.json in {path}")
    base_acc = np.array([r["baseline"]["acc_avg"] for r in runs])
    base_loss = np.array([r["baseline"]["loss"] for r in runs])
    keys = sorted({(x["bits"], x["keep"], x["scale"]) for r in runs for x in r["results"]},
                  key=lambda k: (k[2], k[1], -k[0]))
    table, status = [], {}
    for k in keys:
        cur = [next(x for x in r["results"] if (x["bits"], x["keep"], x["scale"]) == k) for r in runs]
        drop = base_acc - np.array([c["acc_avg"] for c in cur])      # + = worse
        rise = np.array([c["loss"] for c in cur]) - base_loss        # + = worse
        sd = lambda v: float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        collapsed = bool(drop.mean() > ACC_COLLAPSE or rise.mean() > LOSS_COLLAPSE)
        status[k] = collapsed
        table.append((k, drop.mean(), sd(drop), rise.mean(), sd(rise), collapsed,
                      bool(abs(drop.mean()) > sd(drop)), bool(abs(rise.mean()) > sd(rise))))

    gates = []
    if len(runs) < 3: gates.append(f"only {len(runs)} seed(s), need >= 3")
    srcs = {r["meta"]["text_source"] for r in runs}
    if srcs != {"tinystories"}: gates.append(f"text source {sorted(srcs)} is not real TinyStories")
    if base_acc.mean() < 0.5: gates.append(f"baseline recall {base_acc.mean():.2f} too low to be informative (chance 0.016)")

    def get(bits, keep, scale="dynamic"): return status.get((bits, keep, scale))
    claims = [("4-bit within thresholds (dynamic, no kept channels)", get(4, 0) is False),
              ("2-bit collapses (dynamic, no kept channels)", get(2, 0) is True)]
    for kp in sorted({k[1] for k in keys if k[1] > 0}):
        claims.append((f"2-bit + top {kp:.0%} channels in FP stays within thresholds", get(2, kp) is False))

    L = [f"# Summary: {path}", "",
         f"seeds: {len(runs)}  |  baseline recall {base_acc.mean():.3f} ± {base_acc.std(ddof=1) if len(runs)>1 else float('nan'):.3f}"
         f"  |  baseline loss {base_loss.mean():.4f} nats/byte", "",
         "| bits | keep | scale | recall drop (pts) | ±sd | loss rise | ±sd | collapse | drop>sd | rise>sd |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for (k, dm, ds, rm, rs, c, e1, e2) in table:
        L.append(f"| {k[0]} | {k[1]:.2f} | {k[2]} | {100*dm:.1f} | {100*ds:.1f} | {rm:.4f} | {rs:.4f} | "
                 f"{'YES' if c else 'no'} | {'yes' if e1 else 'no'} | {'yes' if e2 else 'no'} |")
    L += ["", "## Claims in FINDING.md", ""] + [f"- {'holds' if ok else 'DOES NOT HOLD / not tested'}: {t}" for t, ok in claims]
    L += ["", "## Verdict", "",
          ("PRELIMINARY OBSERVATION, not a finding. Gates failed: " + "; ".join(gates)) if gates else
          "All gates passed. Whether each claim counts as a finding also needs the effect-size columns above (drop>sd / rise>sd) and the thresholds to be excluded by the paired difference; check those by hand before writing it up."]
    text = "\n".join(L)
    print(text)
    open(os.path.join(path, "summary.md"), "w").write(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
