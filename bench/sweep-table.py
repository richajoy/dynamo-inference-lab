#!/usr/bin/env python3
"""Table from one bench/sweep.sh run. Runs ON the node (stdlib only):
  python3 bench/sweep-table.py /var/lib/bench-results/<run-dir> [gpus]
gpus = GPUs the deployment used (default 1) -> output tok/s per GPU.
Per shape and concurrency: req/s, input tok/s, output tok/s, output tok/s per GPU, tok/s per user,
TTFT p50/p99, ITL (=TPOT) avg/p99, E2E latency avg. Values come from AIPerf's profile_export_aiperf.json.
"""
import json, pathlib, sys

run = pathlib.Path(sys.argv[1]); gpus = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

def find(o, key):
    if isinstance(o, dict):
        if key in o and isinstance(o[key], dict): return o[key]
        for v in o.values():
            r = find(v, key)
            if r: return r
    if isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r: return r
    return None

def g(prof, key, stat="avg"):
    m = find(prof, key)
    return m.get(stat) if m else None

def f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"

rows = []
for p in sorted(run.glob("*/c*/profile_export_aiperf.json"), key=lambda p: (p.parts[-3], int(p.parts[-2][1:]))):
    prof = json.load(open(p))
    shape, c = p.parts[-3], int(p.parts[-2][1:])
    out_tps = g(prof, "output_token_throughput")
    rows.append({
        "shape": shape, "c": c,
        "req_s": g(prof, "request_throughput"),
        "in_tps": (g(prof, "request_throughput") or 0) * (g(prof, "input_sequence_length") or 0),
        "out_tps": out_tps, "out_tps_gpu": out_tps / gpus if out_tps else None,
        "tps_user": g(prof, "output_token_throughput_per_user"),
        "ttft_p50": g(prof, "time_to_first_token", "p50"), "ttft_p99": g(prof, "time_to_first_token", "p99"),
        "itl_avg": g(prof, "inter_token_latency"), "itl_p99": g(prof, "inter_token_latency", "p99"),
        "e2e_avg": (g(prof, "request_latency") or 0) / 1000,
        "errors": (find(prof, "error_request_count") or {}).get("avg", 0), "ok": (find(prof, "request_count") or {}).get("avg"),
    })

print(f"run: {run.name}  gpus={gpus:g}")
print("| shape | c | req/s | input tok/s | output tok/s | out tok/s/GPU | tok/s/user | TTFT p50 ms | TTFT p99 ms | ITL avg ms | ITL p99 ms | E2E avg s | ok/err |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['shape']} | {r['c']} | {f(r['req_s'],2)} | {f(r['in_tps'],0)} | {f(r['out_tps'],0)} | {f(r['out_tps_gpu'],0)} | "
          f"{f(r['tps_user'])} | {f(r['ttft_p50'],0)} | {f(r['ttft_p99'],0)} | {f(r['itl_avg'],2)} | {f(r['itl_p99'],2)} | {f(r['e2e_avg'],2)} | {f(r['ok'],0)}/{f(r['errors'],0)} |")
out = pathlib.Path("results/tables"); out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / f"{run.name}.table.json", "w"), indent=2)
