\# QUESTION.md



\## 1. The question

How many bits can a small SSM’s recurrent hidden state be quantized to (post-training) before associative recall collapses, and which channels matter most?



Collapse is defined as average recall accuracy falling more than 10 percentage points below the full-precision baseline, or language-modeling loss rising more than 0.05 nats, on the evaluation tasks below.



Default quantization for round one: uniform asymmetric min-max quantization with per-tensor scales, round-to-nearest, applied to the state after every recurrent update (float32 arithmetic otherwise). Scales are dynamic (recomputed from the current state at each step). Static scales (fixed from a calibration set) are treated as an explicit experimental variable.



Channel importance is measured by ranking channels according to the drop in recall when each is ablated (zeroed) on a small calibration set that is held out from the evaluation set; we then keep the top-k channels at full precision while quantizing the rest.



\## 2. Why it matters

A smaller, lower-precision state cuts memory bandwidth and KV-cache-like costs for long-context inference, enabling cheaper on-device SSMs and clarifying what information the continuous state actually stores.



\## 3. What you expect

8-bit quantization will be essentially free (≤2-point recall drop and ≤0.02 nats loss increase).  

4-bit will cost roughly 5–15 points of recall while increasing loss by \~0.03–0.08 nats.  

2-bit and below will exceed the collapse thresholds on the longer distances.



\## 4. What would surprise you

A small subset of channels (top 10–20 %) carrying nearly all the long-range memory, or 2-bit quantization remaining within the collapse thresholds after only simple post-training scale calibration (no quantization-aware training).



\## 5. Out of scope

Models larger than \~20 M parameters; quantization-aware training; anything beyond TinyStories plus controlled synthetic associative-recall / name-recall tasks at fixed distances of 64, 256 and 1 024 tokens; continuous-time formulations, multi-scale hierarchies, or hardware kernels. Simple post-training calibration of quantization scales is allowed; full QAT is not.

