# Exercise 2: Aggregated serving on one H100, from one user to sixty-four

**Goal.** Measure what one H100 running one vLLM engine delivers for different kinds of traffic as load rises, and
explain every curve with the step-time model from the [foundations](00-foundations.md). This run is the baseline
every later exercise is compared with.

**What you will be able to answer after this exercise**
- Why per-GPU throughput rises ~43x from 1 to 64 concurrent users while each user's stream slows by only ~30%.
- Why long prompts hurt everyone else's stream, and by how much.
- Where the floor on time-to-first-token comes from (it is not the GPU).
- What a faster GPU buys, and why it is different for prompts and for generation.

---

## Setup

| Item | Value |
|---|---|
| Deployment | [deploy/agg-qwen3-8b.yaml](../deploy/agg-qwen3-8b.yaml): one Dynamo frontend and one vLLM worker on one whole H100 SXM 80GB. No router choice (one worker) |
| Model | Qwen/Qwen3-8B, BF16 weights and KV cache |
| Engine | vLLM 0.28 via Dynamo 1.5.1. `--gpu-memory-utilization 0.90`, `--max-model-len 8192`, `--max-num-seqs 64`, block 16. Chunked prefill, prefix caching and CUDA graphs on (defaults). Chunk budget 8,192 (the vLLM default on this GPU) |
| KV capacity | 388,080 tokens (engine log) |
| Load | [bench/sweep.sh](../bench/sweep.sh), closed loop at concurrency 1, 4, 16, 64; prefill-heavy (3,000 in / 128 out) and decode-heavy (256 in / 1,024 out). [Harness details](01-stack-and-deployment.md#5-the-benchmark-harness) |

"Aggregated" means each GPU does both prefill and decode for its own requests, interleaved step by step by chunked
prefill. It is the default way to serve, and the simplest.

## Results

![Six panels for one H100: TTFT p50 and p99, TPOT, and output tokens per second per GPU and per user, against
concurrency, for prefill-heavy (top row) and decode-heavy (bottom row) traffic](../figures/agg-h100.png)

**How to read this chart.** Top row: prefill-heavy traffic. Bottom row: decode-heavy. Left: time to first token, log
scale. Middle: time per output token. Right: total output tokens per second the GPU produced (green, left axis) and the
speed one user's stream sees (blue, right axis). X axis: requests kept in flight.

### Decode-heavy: batching is the whole game

- **One user:** a token every 6.85 ms, 146 tokens/s for that user, 145 tokens/s for the GPU. The GPU is reading 15 GB
  of weights to produce one token per step. Section 4 of the foundations predicts this from bandwidth alone.
- **64 users:** a token every 9.94 ms per user (100 tokens/s each, 31% slower), while the GPU produces **6,278
  tokens/s, 43x more**. Every step still reads the weights once; it now produces 64 tokens with that read. The only
  added cost is each request's own KV cache, ~7 GB per step at this context length, which the model predicts within
  1%.
- **Time to first token** stays at **43 ms** up to 4 users and reaches 249 ms at 64, because a new prompt waits for
  the step in flight and shares the next one.

At short context, adding users is almost free for the users and enormously valuable for the operator. That is why
serving systems batch, and why concurrency is the first knob.

### Prefill-heavy: interference and queueing

- **One user:** TTFT 79 ms. A 3,000-token prompt is ~44 TFLOP; the GPU spends ~70 ms on it (trace in
  [exercise 7](07-profiling.md)), and the remaining ~9 ms is software overhead. Then 127 tokens at 7.0 ms each.
- **16 users:** TTFT p50 591 ms, TPOT 13.8 ms. With an 8,192-token budget a single engine step can carry about 2.7 of
  these prompts. Our explanation, consistent with the numbers but not traced: they all get their first token when that long
  step ends, so the median TTFT reflects batching prompts together, not a full GPU.
- **64 users:** TTFT p50 617 ms but **p99 4.7 s**; TPOT **43.6 ms** (24 tokens/s per user); 9.6 requests/s and 28,700
  prompt tokens/s. The GPU drew 686 W on average against its 700 W limit.

The TPOT jump from 7 to 43.6 ms is not because decode got slower. The [interference chart](00-foundations.md#5-mixing-prefill-and-decode-in-one-step-interference)
splits it: the median gap between tokens is 17 ms, within 3% of the decode-only step the model predicts for ~53 requests
with ~3,000-token contexts, while one gap in ten is ~200 ms, about the length of a step that also carries an 8,192-token prompt chunk.
**61% of the TPOT at c64 lies above the median gap**, which we attribute to decoding users waiting for other users'
prompts. The fix options are the
subject of later exercises: smaller chunks ([exercise 6](06-engine-tuning.md)), or separate GPUs for prefill and
decode ([exercise 4](04-pd-disaggregation.md)).

KV capacity is not the limit here: 64 requests of 3,128 tokens need ~200,000 of the 388,080 available tokens. The
tail TTFT is queueing for compute.

### The 43 ms floor on time to first token

A 256-token prompt is a few milliseconds of GPU work, yet no request ever got its first token in less than ~43 ms, on
this GPU or the A100. This minimum covers HTTP handling, tokenization, the Dynamo frontend hop to the worker,
scheduling, the CPU-launched prefill step and the first decode step (AIPerf saw the first two tokens arrive together). We
attribute it by elimination, not by profiling each hop. The practical point: **for short prompts, the serving stack,
not the GPU, sets the minimum TTFT.** It is a floor, not a constant added to every request: a 3,000-token prompt's
79 ms TTFT is ~70 ms of GPU work plus ~9 ms. It also caps what prefix caching can do for TTFT ([exercise 3](03-kv-aware-routing.md)).

### The throughput-interactivity frontier

![Output tokens per second per GPU against tokens per second per user for decode-heavy traffic: H100 BF16, H100 with
FP8 weights, H100 with the combined tuned configuration, and the A100 BF16 for reference; points labelled by
concurrency](../figures/frontier-decode.png)

**How to read this chart.** Each line is one configuration; each dot is one concurrency level. Up is more throughput
per GPU, right is a faster stream per user. A line that sits entirely up and to the right of another dominates it.

**What it shows**
- Moving along a line is the concurrency dial: from c1 to c64 the H100 BF16 line gains 43x throughput for 31% of
  per-user speed. There is no free point; you choose one.
- Moving to a different line is a configuration change. The tuned H100 lines ([exercise 6](06-engine-tuning.md))
  sit above and to the right of plain BF16; the A100 line sits below and to the left of everything.

### Energy per token

| Decode-heavy | c1 | c64 |
|---|---|---|
| Board power | 426 W | 618 W |
| Output tokens per joule | 0.34 | 10.16 |

Power rises 45% while output rises 43x, so **energy per token falls 30x** (about 98 mJ per token at c64). An
under-loaded GPU is the most expensive way to serve a token.

<details><summary>Full H100 tables</summary>

Prefill-heavy (3,000 / 128):

<!-- BEGIN GENERATED: agg-prefill-heavy -->
| c | req/s | TTFT p50 ms | TTFT p99 ms | TPOT ms | out tok/s per GPU | in tok/s per GPU | tok/s per user | avg power W | output tok per joule | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.03 | 79 | 83 | 7.00 | 132 | 3,095 | 142.9 | 447 | 0.30 | 1 | 32/0 |
| 4 | 3.22 | 271 | 295 | 7.95 | 412 | 9,664 | 126.6 | 515 | 0.80 | 4 | 32/0 |
| 16 | 6.84 | 591 | 1,179 | 13.76 | 875 | 20,507 | 74.1 | 596 | 1.47 | 16 | 64/0 |
| 64 | 9.56 | 617 | 4,727 | 43.60 | 1,223 | 28,681 | 24.4 | 686 | 1.78 | 64 | 256/0 |
<!-- END GENERATED -->

Decode-heavy (256 / 1,024):

<!-- BEGIN GENERATED: agg-decode-heavy -->
| c | req/s | TTFT p50 ms | TTFT p99 ms | TPOT ms | out tok/s per GPU | in tok/s per GPU | tok/s per user | avg power W | output tok per joule | avg in flight | ok/err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.14 | 43 | 44 | 6.85 | 145 | 36 | 146.0 | 426 | 0.34 | 1 | 32/0 |
| 4 | 0.55 | 43 | 44 | 7.01 | 567 | 142 | 142.6 | 435 | 1.30 | 4 | 32/0 |
| 16 | 2.03 | 97 | 111 | 7.59 | 2,083 | 521 | 131.8 | 478 | 4.36 | 16 | 64/0 |
| 64 | 6.13 | 249 | 395 | 9.94 | 6,278 | 1,570 | 100.6 | 618 | 10.16 | 64 | 256/0 |
<!-- END GENERATED -->

"avg in flight" is the concurrency the client actually held; every point here held it.
</details>

## The same experiment on an A100

We ran the identical deployment and harness on an A100 SXM 40GB first. The first-principles prediction is that decode
gains the bandwidth ratio (2.15x) and saturated prefill gains the compute ratio (3.17x).

![H100 over A100 ratio at each concurrency for prefill-heavy requests per second and decode-heavy output tokens per
second per GPU, against reference lines at the 3.17x compute ratio and the 2.15x bandwidth ratio](../figures/scaling-ratio.png)

**What it shows**
- Decode-heavy throughput: 1.93x to 2.06x at every concurrency, just under the bandwidth ratio, as predicted.
- Prefill-heavy requests/s: about 2x at one user (where most of each request's life is decode), rising to ~3x at 64
  users, where the GPU is saturated with prompt arithmetic.
- **A trap in the comparison.** vLLM picks its default chunk budget by GPU: 8,192 on the H100 80GB, 2,048 on the A100
  40GB (vLLM 0.28, `get_batch_defaults`). The same manifest therefore ran two different engine configurations. The
  dashed line re-runs the A100 at 8,192 for a like-for-like comparison (up to c16, where it gives 2.03x to 2.38x).
  We only found this by reading the engine logs; the manifests were identical.

The A100's prefill-heavy c64 point shows what saturation looks like. Its median TTFT was **7.6 s** against the H100's
0.6 s, for two compounding reasons:
1. **Compute:** from c16 to c64 its prompt throughput rose only 10%. It was already at its knee.
2. **KV capacity:** its 40 GB left room for 129,344 tokens of KV, about 41 of these 3,128-token requests. The engine
   ran at most 42 at once; the rest waited. By Little's law, that wait is the TTFT.

The H100 has ~3x the prefill rate and 3x the KV capacity, so 64 requests sit below both limits.

<details><summary>A100 vs H100 table</summary>

<!-- BEGIN GENERATED: a100-vs-h100 -->
| shape | c | A100 (req/s or out tok/s) | H100 | ratio | A100 TPOT ms | H100 TPOT ms |
|---|---|---|---|---|---|---|
| prefill-heavy | 1 | 0.5 | 1.0 | 2.04x | 13.70 | 7.00 |
| prefill-heavy | 4 | 1.5 | 3.2 | 2.16x | 16.86 | 7.95 |
| prefill-heavy | 16 | 2.8 | 6.8 | 2.45x | 37.54 | 13.76 |
| prefill-heavy | 64 | 3.1 | 9.6 | 3.12x | 93.99 | 43.60 |
| decode-heavy | 1 | 74.2 | 145.2 | 1.96x | 13.44 | 6.85 |
| decode-heavy | 4 | 293.2 | 567.0 | 1.93x | 13.57 | 7.01 |
| decode-heavy | 16 | 1,059 | 2,083 | 1.97x | 14.89 | 7.59 |
| decode-heavy | 64 | 3,042 | 6,278 | 2.06x | 20.53 | 9.94 |
<!-- END GENERATED -->
</details>

## What it means

| Your traffic | What sets its performance on one GPU | First lever |
|---|---|---|
| Chat, code generation (short prompts, long answers) | Bandwidth: TPOT follows bytes per step. Throughput follows batch size | Run at the highest concurrency your TPOT target allows; then cut bytes per step (FP8, [exercise 6](06-engine-tuning.md)) |
| RAG, summarization (long prompts, short answers) | Compute for prefill, plus interference on everyone's TPOT | Chunk budget or disaggregation ([exercise 6](06-engine-tuning.md), [exercise 4](04-pd-disaggregation.md)) |
| Long-context agents | KV capacity first, then bandwidth (long contexts make KV reads dominate) | FP8 KV cache, admission limits matched to KV capacity |
| Batch or offline | Throughput per joule | Maximum concurrency the KV cache allows |

Two general lessons:
- **Interference is visible only in the token-gap distribution.** Mean TPOT hides it; the median
  and the 90th percentile of the gap between tokens expose it.
- **Read the engine's startup log for every run.** Defaults that depend on the GPU changed our comparison silently.

## Caveats

- One run per point, one model size (8B), synthetic prompts with no prefix reuse.
- Closed-loop load. Open-loop bursts would queue longer past the knee.
- p99 at c1 to c16 is the single slowest request.
- The A100 runs had no power telemetry (it was wired up later), so energy is H100 only.
- The A100 comparison at c64 mixes chunk budgets; there is no like-for-like c64 point.

## Reproduce

```bash
kubectl -n dynamo-system apply -f deploy/agg-qwen3-8b.yaml
kubectl -n dynamo-system cp bench/sweep.sh aiperf:/tmp/sweep.sh
kubectl -n dynamo-system exec -it aiperf -- bash /tmp/sweep.sh h100-agg-baseline
# to remove the chunk-budget difference between GPUs, add  --max-num-batched-tokens 2048  (or 8192) to the worker args
python3 figures/make_figures.py      # rebuilds every chart and table from results/
```

Next: [Exercise 3, KV-aware routing](03-kv-aware-routing.md).
