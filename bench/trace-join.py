#!/usr/bin/env python3
"""Per-request analysis of one A/B run from the Dynamo frontend request trace. Runs ON the node (stdlib only).

  python3 bench/trace-join.py /var/lib/request-trace/kvr <t0_ms> <t1_ms> [label]
  (t0/t1 = "window_ms" from the run's summary.json)

Why: the aggregate metrics mix units (router = blocks/blocks averaged, vLLM = tokens/tokens summed). The trace has,
per request: what the router PREDICTED (kv_hit_rate, kv modes only), what the engine ACTUALLY reused
(cached_tokens), which worker served it, the TTFT split (queue vs prefill), and the prompt's block hashes.

Outputs:
  actual hit       = sum(cached_tokens) / sum(input_tokens)            (comparable to vLLM's counter ratio)
  block hit        = floor(cached/16) / ceil(input/16) per request    (same unit as the router's prediction)
  prediction error = predicted - actual block hit, per request        (>0: router believed blocks that were evicted)
  prefix affinity  = per prefix (first block hash), share of its requests on its most-used worker
                     (~0.5 with 2 workers = no stickiness; ~1.0 = every prefix pinned to one worker)
  TTFT by outcome  = p50 TTFT / prefill / queue for cache hits (>= half the prompt cached) vs misses
"""
import json, math, statistics as st, sys
from collections import Counter, defaultdict

path, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
label = sys.argv[4] if len(sys.argv) > 4 else ""
BS = 16

recs = []
for line in open(path):
    try:
        ev = json.loads(line)["event"]
    except (ValueError, KeyError):
        continue
    r = ev.get("request") or {}
    if ev.get("event_type") != "request_end" or not (t0 <= ev.get("event_time_unix_ms", 0) <= t1):
        continue
    if not r.get("input_tokens"):
        continue
    recs.append(r)

if not recs:
    sys.exit(f"no request_end records in window {t0}..{t1}")

def p(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None

inp = sum(r["input_tokens"] for r in recs)
cached = sum(r.get("cached_tokens") or 0 for r in recs)
blk = [math.floor((r.get("cached_tokens") or 0) / BS) / math.ceil(r["input_tokens"] / BS) for r in recs]
pred = [(r.get("kv_hit_rate"), b) for r, b in zip(recs, blk) if r.get("kv_hit_rate") is not None]
workers = Counter(str((r.get("worker") or {}).get("decode_worker_id")) for r in recs)

by_prefix = defaultdict(Counter)
for r in recs:
    hs = (r.get("replay") or {}).get("input_sequence_hashes") or []
    if hs:
        by_prefix[hs[0]][str((r.get("worker") or {}).get("decode_worker_id"))] += 1
aff_num = sum(max(c.values()) for c in by_prefix.values())
aff_den = sum(sum(c.values()) for c in by_prefix.values())

hit = [r for r in recs if (r.get("cached_tokens") or 0) >= r["input_tokens"] / 2]
miss = [r for r in recs if (r.get("cached_tokens") or 0) < r["input_tokens"] / 2]
def split(rs):
    return {"n": len(rs), "ttft_p50": p([r.get("ttft_ms") for r in rs], .5), "ttft_p90": p([r.get("ttft_ms") for r in rs], .9),
            "prefill_p50": p([r.get("prefill_time_ms") for r in rs], .5), "queue_p50": p([r.get("prefill_wait_time_ms") for r in rs], .5)}

out = {
    "label": label, "requests": len(recs), "requests_by_worker": dict(workers),
    "actual_hit_tokens": round(cached / inp, 4),
    "actual_hit_blocks_mean": round(st.mean(blk), 4),
    "prefixes_seen": len(by_prefix), "prefix_affinity": round(aff_num / aff_den, 4) if aff_den else None,
    "ttft_all": split(recs), "ttft_cache_hit": split(hit), "ttft_cache_miss": split(miss),
}
if pred:
    err = [a - b for a, b in pred]
    out["predicted_hit_blocks_mean"] = round(st.mean(a for a, _ in pred), 4)
    out["prediction_error_mean"] = round(st.mean(err), 4)
    out["prediction_abs_error_mean"] = round(st.mean(abs(e) for e in err), 4)
    out["overpredicted_by_gt_0.25"] = round(sum(e > .25 for e in err) / len(err), 4)
else:
    out["predicted_hit_blocks_mean"] = None   # round-robin: no index, no prediction
print(json.dumps(out, indent=2))
