# autolab

An automated research lab: agents plan, run, review and write up small ML experiments.

Each round starts from two fixed documents, written before any run: the question, and the rule for what counts as a finding. The result is judged against that rule, so it can't be reframed after the fact.

## Round 1: quantizing a small SSM's recurrent state

**Question** ([QUESTION.md](QUESTION.md)): how many bits can a small state-space model's recurrent state be quantized to, after training, before associative recall collapses, and which channels matter most?

**What counts as a finding** ([FINDING.md](FINDING.md)): average recall stays within 10 percentage points of full precision and language-modeling loss within 0.05 nats, over at least 3 seeds, with an effect larger than the seed-to-seed spread.

**Status:** the question, the finding rules and the experiment runner are written. The runner has not been run end to end yet, so there are no results.

## Layout

| Path | What it holds |
|---|---|
| [QUESTION.md](QUESTION.md) | The question, expectations, what would be a surprise, and what is out of scope |
| [FINDING.md](FINDING.md) | The metric, seeds, baseline and effect-size rule a result must meet |
| [runner/](runner/) | The experiment: recall tasks, a small selective SSM, the quantizer, evaluation, training and aggregation |
| `agents/`, `instrument/`, `notebook/` | Reserved for later rounds (empty) |

## Run

From the `autolab` folder, with `pip install torch numpy`:

```
python -m runner.selftest
python -m runner.run --name smoke --seeds 0 --train-steps 50 --d-model 64 --layers 2 --bits 8 4 2 --keep 0 0.15 --eval-n 16 --calib-n 4 --lm-eval-batches 2
python -m runner.aggregate runs/smoke
```

Smoke runs use generated text and can never count as a finding. [runner/README.md](runner/README.md) lists the design decisions and known limits.

## License

[MIT](LICENSE)
