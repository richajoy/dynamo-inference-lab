# Findings, corrections and open questions

This page collects what the project established, what it got wrong along the way and how that was caught, and what
we would run next. Each item links to the part with the evidence.

- [What we set out to prove, and the verdict](#what-we-set-out-to-prove-and-the-verdict)
- [Findings](#findings)
- [What we got wrong first](#what-we-got-wrong-first)
- [Open questions and next experiments](#open-questions-and-next-experiments)
- [Lessons for anyone benchmarking inference](#lessons-for-anyone-benchmarking-inference)

---

## What we set out to prove, and the verdict

| Claim | Verdict | Evidence |
|---|---|---|
| Decode is memory-bandwidth-bound, so its step time follows bytes read | **Holds.** One fitted constant (66.5% of peak bandwidth) predicts every measured decode step within 3% | [Foundations §4](00-foundations.md#4-a-decode-step-time-model-that-predicts-our-measurements) |
| Prefill is compute-bound | **Holds.** A 3,000-token prompt keeps the GPU busy ~70 ms, ~64% of peak FLOPS; saturated prefill scaled with the compute ratio from A100 to H100 (chunk budgets differ at c64) | [Part 7](07-profiling.md), [Part 2](02-aggregated-serving.md#the-same-experiment-on-an-a100) |
| Mixing prefill into decode steps hurts decoders | **Holds, and it is large.** At 64 concurrent prompt-heavy requests, 61% of mean TPOT lies above the median token gap, which we attribute to steps carrying prompt chunks | [Foundations §5](00-foundations.md#5-mixing-prefill-and-decode-in-one-step-interference) |
| KV-aware routing beats round-robin on shared-prefix traffic | **Holds only after tuning.** Defaults: 0.69 against 0.55 hit rate on A100 (0.64 against 0.54 on H100). With `--router-prefill-load-scale 10`: 0.90 (0.89 on H100) | [Part 3](03-kv-aware-routing.md) |
| Higher hit rates mean lower TTFT | **Depends on the miss cost.** Large gains where a miss is expensive; none at the median where a miss costs less than the fixed per-request path | [Part 3](03-kv-aware-routing.md#why-hits-stopped-mattering-for-median-ttft-on-the-h100) |
| Disaggregation removes interference | **Holds over NVLink:** TPOT halved on prompt-heavy load at 64 concurrent requests. **Fails on standard pods:** one GPU per pod makes UCX fall back to TCP, and throughput drops up to 18x (prompt-heavy, 64 users) | [Part 4](04-pd-disaggregation.md) |
| Disaggregation beats aggregation on the same GPUs | **Not with two GPUs.** A fixed 1:1 split saturates whichever pool the traffic leans on; two aggregated workers won throughput on every shape at c64 | [Part 4](04-pd-disaggregation.md#3-pd-against-two-aggregated-workers-on-the-same-two-gpus) |
| The planner moves GPUs between pools as traffic shifts | **Yes, by deleting and cold-starting pods, about 2 to 3 minutes per move,** and it can move capacity the wrong way when the KV transfer is slow | [Part 5](05-autoscaling-and-mig.md) |
| Engine settings can substitute for hardware | **Partly.** One tuned H100 matched the throughput of a 1:1 prefill/decode pair on two | [Part 6](06-engine-tuning.md) |

## Findings

### Performance mechanisms

1. **Decode step time is predictable from bytes alone.** `step = (15.14 GB of weights + batch x context x 147,456 B) /
   2.23 TB/s` matched every decode measurement within 3%, including a different workload than the one it was fitted
   on. The 66.5% bandwidth efficiency is a property of this model, engine and GPU, and the target of kernel work, not a
   ceiling. Consequences: batching amortizes weight reads but not KV reads, and above ~350 tokens of average context no
   batch size makes decode compute-bound. ([Foundations](00-foundations.md))
2. **Interference hides in the mean.** Prompt-heavy traffic at c64 had a 17 ms median token gap (the decode-only step)
   and a ~200 ms 90th percentile, close to the ~190 ms predicted for a step carrying a full prompt chunk. We read it as
   compute contention. Mean TPOT (43.6 ms) blends the two; the gap distribution separates them. ([Foundations §5](00-foundations.md#5-mixing-prefill-and-decode-in-one-step-interference))
3. **The chunk budget moves waiting, it does not remove it.** At saturation, 512 cut TPOT 60% and raised median TTFT
   7.7x. Below saturation, 512 cut TTFT p50 74% and raised requests/s 9%, with TPOT unchanged. ([Part 6](06-engine-tuning.md#tokens-per-step-the-chunk-budget))
4. **FP8 KV pays where capacity binds; FP8 weights pay where bandwidth binds.** Long-context c64: FP8 KV +37%
   requests/s. Decode-heavy c16: FP8 weights -29% TPOT, less than the -41% the byte count predicts (two candidates:
   the activation quantize step and skinny FP8 decode GEMMs). ([Part 6](06-engine-tuning.md))
5. **Speculative decoding's gain shrinks as the batch grows,** as the roofline predicts: n-gram gave 2.1x TPOT at one
   user and 1.7x at 64, on synthetic text that flatters acceptance. ([Part 6](06-engine-tuning.md#tokens-emitted-per-step-speculative-decoding))

### Routing

6. **Dynamo's default KV-router weighting barely beats round-robin.** At `prefill_load_scale` 1, a full prefix miss
   costs about the same as one extra running request, so routine load imbalance spills requests to cold workers. Scale
   10 keeps prefixes sticky: 0.90 hit rate, the same on two GPU generations. ([Part 3](03-kv-aware-routing.md#why-the-default-router-barely-beats-round-robin))
7. **Approximate mode with the default TTL over-predicts its own hit rate by 26 points** under cache pressure; LRU
   simulation tracks reality. ([Part 3](03-kv-aware-routing.md#approximate-mode-ttl-over-predicts-lru-tracks-reality))
8. **Dynamo's request trace records the router's prediction as `cached_tokens` in KV modes,** not the engine's result,
   so it cannot be used to check the router per request. ([Part 3](03-kv-aware-routing.md#the-request-trace-records-the-routers-guess-not-the-engines-result))
9. **Router-side queueing cut TTFT p90 2.4x under overload**, at ~34 ms average time in the router. ([Part 3](03-kv-aware-routing.md#under-overload-router-side-queueing))

### Disaggregation and autoscaling

10. **Single-node disaggregation on standard Kubernetes pods silently moves KV over TCP:** one GPU per pod hides the
    peer GPU from CUDA, so UCX falls back with no warning. ([Part 4](04-pd-disaggregation.md#1-check-the-transport-first-the-default-pod-spec-falls-back-to-tcp))
11. **Once the link is fast, descriptor posting limits KV transfers.** Under load the free-block list fragments and a
    6,000-token transfer becomes 13,500 memory spans of 64 KiB; posting them took 55% of each transfer and cut
    bandwidth to ~9.6 GB/s. The mechanism is reported upstream for an MLA model (vLLM issue #55434); we measured it on a
    GQA model with vLLM's default one-region-per-layer layout. Block size 64 made transfers ~3.5x faster.
    ([Part 4](04-pd-disaggregation.md#2-with-a-fast-link-the-copy-list-becomes-the-bottleneck))
12. **Admit no more decode requests than the KV cache holds.** A decode worker admitting 64 long requests where 55 fit
    stalled at 100% KV, failed its health check and was restarted (seen once). ([Part 4](04-pd-disaggregation.md#4-a-decode-worker-given-more-kv-than-it-holds))
13. **The planner cannot tell transfer back-pressure from decode load.** Over a slow link, requests waiting for their
    KV appear as queued decode tokens, and the planner moved a slice from prefill to decode during prompt-heavy load.
    Moves take about 2 to 3 minutes and single quiet samples caused flapping. ([Part 5](05-autoscaling-and-mig.md))

### Operations and measurement

14. **The same manifest ran different engine configurations on different GPUs.** vLLM 0.28 sets the default chunk
    budget by GPU (8,192 on an H100 80GB, 2,048 on an A100 40GB). Only the engine's startup log shows it.
    ([Part 2](02-aggregated-serving.md#the-same-experiment-on-an-a100))
15. **`GPU_UTIL` reads ~100% at any load and reports nothing on MIG;** DCGM's default 30 s interval hides short runs.
    ([Part 1](01-stack-and-deployment.md#monitoring-and-two-instrument-lessons), [Part 5](05-autoscaling-and-mig.md#2-monitoring-goes-partly-blind))
16. **Dynamo's "NVLink Bandwidth" dashboard panel always reads ~0,** because it applies `rate()` to a DCGM field that is
    already a rate. ([Part 4](04-pd-disaggregation.md#5-dashboards-during-the-pd-tests))
17. **A fresh Dynamo platform install fails until KAI's webhook is up,** and **a frontend can be Ready before any
    worker is routable.** Both are handled in the scripts. ([Part 1](01-stack-and-deployment.md#problems-we-hit-and-the-fixes-now-in-the-scripts))

## What we got wrong first

Writing this project up forced a re-check of every number against the raw exports and logs. These are the claims we
made during the runs and later corrected. The parts use the corrected versions.

**1. "FP8 does not raise prefill throughput, so prefill is not GEMM-bound." Retracted.** The FP8 run showed the same
9.55 requests/s as BF16 at "c=64", but it held only 47 requests in flight: two sweeps were sharing one load-generator
pod. We caught it with the closed-loop check (requests/s x mean latency / concurrency = 0.73). The FP8 prefill question
is open until re-run with its own client. This is the correction we would most want a reader to remember: the
benchmark client, not the engine, was the bottleneck, and nothing flagged it until we checked.

**2. "FP8 fusion passes are slower on every shape." Weakened.** The fused and plain FP8 runs shared a client and ran at
different effective concurrency at c64 (58 to 63 against 47 to 51 requests in flight), and a bigger batch raises TPOT by itself.
Only the decode-heavy c16 cell compares like with like (+7% TPOT for the fused build, one run).

**3. "The chunk budget makes no difference on the H100 (2,048 against 8,192)." Wrong budget.** The baseline was never
2,048: vLLM defaulted to 8,192 on the H100, so that comparison was a run against its own repeat. The A100 versus H100
comparison also mixed budgets, which made the H100's median TTFT at c16 look "barely better". Like for like it was
2.7x better.

**4. Weights read per decode step: 16.4 GB, so decode reached 72% of peak bandwidth. Revised to 15.14 GB and 66.5%.**
vLLM's loaded size includes the embedding table, which decode does not stream (it looks up one row per token). And
66.5% is the efficiency this stack achieved, not a ceiling on what decode can reach.

**5. "The GPUs were not throttling." Not supported.** During prompt-heavy load at c64 the H100 drew 686 W on average
(median 695 W) against its 700 W limit. We did not collect clocks or throttle reasons for that window, so we cannot say
whether it was power-capped.

**6. A block-64 transfer "took 27.5 ms with 6.7 ms of posting."** That was one 10-second log interval. Averaged over all 13
intervals it was 20.6 ms with 38% posting, but those intervals mix half-size and full-size transfers. Like for like
with block 16 (fully fragmented 6,000-token transfers), block 64 took ~26 ms against ~93 ms, with ~46% posting.

**7. Units.** vLLM's transfer log says "MB" and "MB/s" but divides by 2^20, so they are MiB. The 846 "MB" per
transfer is 846 MiB (887 MB), and it is 376 whole blocks, not 6,001 tokens x 147,456 bytes. The NVLink rate of ~164 GB/s
is in decimal units.

**8. "The tuned H100 delivers 4.2x the A100."** That compared the H100 at 128 concurrent requests with the A100 at 64.
At equal concurrency it is 2.9x (about 2.1x from hardware and 1.4x from tuning).

**9. Speculative decoding "accepted ~3.7 of 4 tokens."** vLLM's acceptance length counts the model's own bonus token,
so the maximum with 4 proposals is 5. Acceptance over whole runs was ~66%.

**10. "Planner moves take 2 min 37 s to 2 min 47 s, and the flips happened during the mixed phase."** Measured from
decision to ready, moves took 1:51 to 2:49 (faster into a free slice, slower when a pod had to be deleted first), and
two of the "mixed phase" moves happened just before and just after it.

**11. "Nsight tracing inflates step time by ~25%." Withdrawn.** That compared a traced step with ~3,000-token contexts
against an untraced step with ~768-token contexts. The KV reads explain the whole difference; tracing overhead is
unmeasured.

## Open questions and next experiments

| Experiment | Question it answers | GPUs |
|---|---|---|
| Re-run FP8 weights with one sweep per client, and add an H100 run at chunk 2,048 | Does FP8 raise saturated prefill throughput? How does 2,048 compare with 8,192 on the H100? | 1 H100, ~30 min |
| Open-loop load (`--request-rate`) around the knee | Capacity at a latency target (for example, the highest request rate with TTFT p99 under 500 ms), which closed-loop sweeps cannot give | 1 H100, ~45 min |
| DCGM clocks, throttle reasons, tensor-pipe and DRAM activity during prompt-heavy c64 | Was the GPU power-capped? How close to the compute and bandwidth limits did each phase get? | 1 H100, ~20 min |
| Disaggregation at 2 prefill : 1 decode | Does a ratio that matches prompt-heavy traffic beat aggregation? | 3+ GPUs |
| Planner with NVLink transfers | Does the wrong-way move disappear when the transport is fast? | 2 H100s |
| Speculative decoding on a real conversation dataset | Realistic acceptance and the crossover concurrency | 1 H100, ~20 min |
| Skewed (hot-prefix) traffic for the router | Where locality weighting hot-spots, and where temperature or credit decay should win | 2 H100s |
| Nsight Compute on one decode GEMM and one attention kernel | Per-kernel bandwidth and roofline position, instead of inference from scaling | 1 H100, ~30 min |
| Repeat the decode stall twice with debug logging | Bug or configuration limit? | 2 H100s |

## Lessons for anyone benchmarking inference

1. **Check that the load generator held the concurrency you asked for.** Requests/s x mean latency / concurrency
   should be ~1. Below ~0.9 the client is the bottleneck.
2. **Count errors in every table.** A failed request finishes fast and can make a run look better.
3. **Verify the data path, not only the configuration.** "KV transfer enabled" said nothing about which transport UCX picked.
4. **Read the engine's startup log for every run.** Defaults depend on the hardware.
5. **Look at distributions, not means.** Mean TPOT hid that half the token gaps were 17 ms or less, but at least one in ten was ~200 ms.
6. **Pin anything that sets capacity** (here, KV cache size) when comparing hardware or routers, or a hit-rate
   difference is really a memory difference.
7. **Say how many samples sit behind a percentile.** The p99 of 32 requests is the maximum.
8. **Write the prediction down before the run.** Most of our corrections were caught because a measurement disagreed
   with arithmetic done first.
