# runner

Runs the experiment in ../QUESTION.md and scores it by ../FINDING.md.

Files: tasks.py (recall tasks, text), model.py (small selective SSM, hookable state),
quant.py (quantizer + numpy reference), evaluate.py (hooks, eval, calibration, channel
ranking), train.py, run.py (per-seed pipeline), aggregate.py (summary + gates), selftest.py.

## Use (from the autolab folder; needs `pip install torch numpy`)
    python -m runner.selftest                       # run first; torch parity tests need torch
    python -m runner.run --name smoke --seeds 0 --train-steps 50 --d-model 64 --layers 2 \
        --bits 8 4 2 --keep 0 0.15 --eval-n 16 --calib-n 4 --lm-eval-batches 2
    python -m runner.run --name round1 --tinystories C:\path\to\TinyStories.txt
    python -m runner.aggregate runs\round1

## Decisions I made (change if you disagree)
- Byte-level vocab, so LM loss is nats per BYTE; the 0.05-nat threshold is applied to that.
- Channel = one index of d_inner (owns d_state entries). Ranking = zero it after every
  update, measure recall drop on a held-out calibration set; top-k taken globally over layers.
- "Per-tensor" = one (lo, hi) per sequence; range always includes 0.
- Recall: 4 key-value bindings, `distance` random filler bytes, then 4 shuffled queries.
- One model trained from scratch per seed (new init + data order); all arms share that model
  and the same eval data. Static scales come from calibration data, not eval data.
- Training: 50/50 mix of recall (distances 16/64/256/1024) and LM, 3000 steps by default.

## Known limits
- Never run end to end yet: the dev shell had no torch. Run selftest first.
- Sequential Python scan: slow at length 1024. Fine for a first round, optimise later.
- Static-scale range is recorded over all channels, including kept ones.
- If baseline recall is low, aggregate.py marks the result uninformative; train longer first.
- aggregate.py checks mean-vs-sd, not the paired-CI test in FINDING.md; check that by hand.
- Smoke runs without --tinystories use fake text and can never count as a finding.
