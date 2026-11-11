"""Data for the autolab SSM-state-quantization question (see ../QUESTION.md).

Token space (byte-level, so no tokenizer dependency):
  0..255   raw bytes (TinyStories text, and filler in recall tasks)
  256..319 recall KEYS   (64)
  320..383 recall VALUES (64)
  384      QUERY marker
  385      PAD
Chance level for recall is 1/64.
"""
import numpy as np

KEY0, NKEYS = 256, 64
VAL0, NVALS = 320, 64
QUERY, PAD = 384, 385
VOCAB = 386
DISTANCES = (64, 256, 1024)


def make_recall_batch(rng, batch, distance, n_pairs=4):
    """Name/multi-query associative recall.

    Layout: [k v]*n_pairs, `distance` random filler bytes, then ([QUERY k v])*n_pairs
    with the queried keys in shuffled order. `distance` is the number of filler
    tokens between the last binding and the first query (later queries are +3 each).
    Returns (inp, tgt) int64 arrays; tgt is -100 except at value positions of queries.
    """
    L = 2 * n_pairs + distance + 3 * n_pairs
    s = np.full((batch, L), PAD, dtype=np.int64)
    m = np.zeros((batch, L), dtype=bool)
    for i in range(batch):
        keys = rng.choice(NKEYS, n_pairs, replace=False)
        vals = rng.integers(0, NVALS, n_pairs)
        pos = 0
        for k, v in zip(keys, vals):
            s[i, pos], s[i, pos + 1] = KEY0 + k, VAL0 + v
            pos += 2
        s[i, pos:pos + distance] = rng.integers(0, 256, distance)
        pos += distance
        for j in rng.permutation(n_pairs):
            s[i, pos], s[i, pos + 1], s[i, pos + 2] = QUERY, KEY0 + keys[j], VAL0 + vals[j]
            m[i, pos + 2] = True
            pos += 3
    inp = s[:, :-1]
    tgt = np.where(m[:, 1:], s[:, 1:], -100)
    return inp, tgt


# ---------------------------------------------------------------- text data
_WORDS = ("the a one day little girl boy dog cat tree sun park friend happy sad "
          "played ran saw found said wanted big small red blue went home together "
          "and but then because very good time story named Lily Tom Ben Mia").split()


def fallback_text(rng, n=2_000_000):
    """Pseudo-text for SMOKE TESTS ONLY. Results using it can never count as a finding."""
    words = rng.choice(_WORDS, size=n // 5)
    return np.frombuffer((" ".join(words)).encode("ascii"), dtype=np.uint8).copy()


def load_text(path, rng):
    """Returns (train, calib, eval, source_tag). Split 80/10/10 by position."""
    if path:
        data = np.fromfile(path, dtype=np.uint8)
        tag = "tinystories"
    else:
        data = fallback_text(rng)
        tag = "synthetic-fallback"
    a, b = int(0.8 * len(data)), int(0.9 * len(data))
    return data[:a], data[a:b], data[b:], tag


def lm_batch(rng, data, batch, length):
    idx = rng.integers(0, len(data) - length - 1, batch)
    x = np.stack([data[i:i + length] for i in idx]).astype(np.int64)
    y = np.stack([data[i + 1:i + length + 1] for i in idx]).astype(np.int64)
    return x, y
