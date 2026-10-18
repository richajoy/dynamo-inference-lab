#!/usr/bin/env python3
"""Build every chart and data table in this repo from the benchmark exports in results/.

Inputs (all committed):
  results/h100/<run>/<shape>/c<N>/profile_export_aiperf.json   AIPerf export per sweep point
  results/h100/<run>/<shape>/c<N>/phases/profiling/gpu_telemetry.json   DCGM samples AIPerf collected (H100 only)
  results/a100/<run>/...                                       same layout for the A100 runs
  results/kv-router/<arm>.json, rf-*.prom.json                 KV-router A/B summaries, router-feature Prometheus windows
  evidence/logs/*.log*                                         NIXL transfer lines, planner replica timeline

Outputs:
  figures/*.png, figures/tables/*.md, and the <!-- BEGIN GENERATED: name --> blocks inside docs/*.md.

Run from anywhere:  python3 figures/make_figures.py   (needs matplotlib and numpy)
"""
import gzip, json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
TAB = OUT / "tables"
DOCS = ROOT / "docs"
RES = ROOT / "results"
LOGS = ROOT / "evidence" / "logs"

SHAPES = {"prefill-heavy": (3000, 128), "decode-heavy": (256, 1024), "long": (6000, 1024)}
COL = {"A100": "#e07b00", "H100": "#4c8c00"}
PAL = ["#4c8c00", "#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#e07b00"]

# ---------------------------------------------------------------- first-principles constants (Qwen3-8B, BF16, H100 SXM)
PEAK_TFLOPS = 989.0            # dense BF16 tensor-core peak, H100 SXM
PEAK_TBS = 3.35                # HBM3 bandwidth, TB/s
RIDGE = PEAK_TFLOPS / PEAK_TBS # ~295 FLOP per byte
W_BYTES = 15.14e9              # weights streamed per decode step: 6.95B non-embedding + 0.62B lm_head params x 2 bytes
KV_TOK = 147_456               # KV bytes per token: 36 layers x 2 (K,V) x 8 KV heads x 128 dims x 2 bytes
F_TOK = 2 * 7.57e9             # FLOPs per decoded token for the dense part (2 x params touched)
ATTN_PER_CTX = 4 * 36 * 32 * 128   # attention FLOPs per token per token of context (QK^T and AV, 32 query heads)

# ---------------------------------------------------------------- run registry
# key -> (node, run directory under results/<node>/, GPUs serving the deployment, human label)
RUNS = {
    "a100-agg":       ("a100", "agg-baseline",          1, "A100 BF16 (vLLM default chunk 2048)"),
    "a100-mbt8192":   ("a100", "mbt8192",               1, "A100 BF16, chunk 8192"),
    "h100-agg":       ("h100", "h100-agg-baseline",     1, "H100 BF16"),
    "agg2":           ("h100", "h100-agg2",             2, "aggregated x2 + KV router"),
    "pd-tcp":         ("h100", "h100-pd-tcp",           2, "P/D 1P1D, KV over TCP"),
    "pd-nvlink":      ("h100", "h100-pd-nvlink",        2, "P/D 1P1D, KV over NVLink"),
    "pd-nvlink-bs64": ("h100", "h100-pd-nvlink-bs64",   2, "P/D 1P1D, NVLink, KV block 64"),
    "base":           ("h100", "h100-agg-base",         1, "BF16 baseline (chunk 8192 = H100 default)"),
    "mbt512":         ("h100", "h100-agg-mbt512",       1, "chunk budget 512"),
    "mbt8k":          ("h100", "h100-agg-mbt8k",        1, "chunk 8192 set explicitly (repeat of baseline)"),
    "fp8":            ("h100", "h100-agg-fp8",          1, "FP8 weights"),
    "fp8-fused":      ("h100", "h100-agg-fp8-fused",    1, "FP8 weights + fusion passes"),
    "kvfp8":          ("h100", "h100-agg-kvfp8",        1, "FP8 KV cache"),
    "spec-ngram":     ("h100", "h100-agg-spec-ngram",   1, "speculative, n-gram k=4"),
    "spec-draft":     ("h100", "h100-agg-spec-draft",   1, "speculative, Qwen3-0.6B draft k=4"),
    "seqs128":        ("h100", "h100-agg-seqs128",      1, "max-num-seqs 128"),
    "best":           ("h100", "h100-agg-best",         1, "combined (FP8 weights + FP8 KV + chunk 1024 + 128 seqs)"),
}
KV_ARMS_A100 = ["rr", "kv", "kv-approx", "kv-pls4", "kv-pls10", "approx-ttl-pls10", "approx-lru-pls10"]
KV_ARMS_H100 = ["h100-rr", "h100-kv", "h100-kv-pls10"]


# ---------------------------------------------------------------- loading
def _index(node):
    out = {}
    for p in (RES / node).rglob("*.json"):
        out[str(p.relative_to(RES / node))] = json.load(open(p))
    return out


IDX = {"a100": _index("a100"), "h100": _index("h100")}


def _telemetry(tel, ngpu):
    """Average power and utilisation over the profiling phase, for the ngpu busiest GPUs (by power)."""
    gpus = []
    for ep in tel["data"]["endpoints"].values():
        for g in ep["gpus"].values():
            m = g["metrics"]
            gpus.append({"power": m.get("nvidia_power_usage", {}).get("avg"),
                         "power_p50": m.get("nvidia_power_usage", {}).get("p50"),
                         "util": m.get("nvidia_gpu_utilization", {}).get("avg")})
    gpus = sorted([g for g in gpus if g["power"] is not None], key=lambda g: -g["power"])[:ngpu]
    if not gpus:
        return {}
    return {"power_w": sum(g["power"] for g in gpus), "power_p50": np.mean([g["power_p50"] or 0 for g in gpus]),
            "gpu_util": np.mean([g["util"] or 0 for g in gpus])}


def point(key, shape, c):
    node, run, ngpu, _ = RUNS[key]
    p = IDX[node].get(f"{run}/{shape}/c{c}/profile_export_aiperf.json")
    if p is None:
        return None
    g = lambda k, s="avg": (p.get(k) or {}).get(s)
    err = sum(e.get("count", 0) for e in (p.get("error_summary") or []))
    r = {"key": key, "shape": shape, "c": c, "ngpu": ngpu,
         "n_ok": int(g("request_count") or 0), "n_err": int(err),
         "req_s": g("request_throughput"),
         "ttft_p50": g("time_to_first_token", "p50"), "ttft_p90": g("time_to_first_token", "p90"),
         "ttft_p99": g("time_to_first_token", "p99"), "ttft_avg": g("time_to_first_token"),
         "itl": g("inter_token_latency"),
         "gap_p50": g("inter_chunk_latency", "p50"), "gap_p90": g("inter_chunk_latency", "p90"),
         "gap_p99": g("inter_chunk_latency", "p99"),
         "out_tps": g("output_token_throughput"), "tps_user": g("output_token_throughput_per_user"),
         "e2e_p50": (g("request_latency", "p50") or 0) / 1000.0,
         "eff_c": (p.get("effective_concurrency") or {}).get("avg")}
    # Closed-loop check (Little's law): with c requests always in flight, req/s x mean E2E = c.
    # A ratio well below 1 means the load generator, not the server, limited the run.
    e2e_avg = (g("request_latency") or 0) / 1000.0
    r["e2e_avg"] = e2e_avg
    r["loop_fill"] = (r["req_s"] * e2e_avg / c) if (r["req_s"] and e2e_avg) else None
    r["out_tps_gpu"] = r["out_tps"] / ngpu if r["out_tps"] else None
    r["in_tps_gpu"] = r["req_s"] * SHAPES[shape][0] / ngpu if r["req_s"] else None
    tel = IDX[node].get(f"{run}/{shape}/c{c}/phases/profiling/gpu_telemetry.json")
    if tel:
        r.update(_telemetry(tel, ngpu))
        if r.get("power_w") and r["out_tps"]:
            r["tok_per_j"] = r["out_tps"] / r["power_w"]
    return r


def series(key, shape, cs=(1, 4, 16, 64, 128)):
    return [p for p in (point(key, shape, c) for c in cs) if p]


def kv_summary(node, run_suffix):
    for k, v in IDX[node].items():
        if k.endswith("summary.json") and k.split("/")[0].endswith(run_suffix) and k.count("/") == 1:
            return v
    return None


def res_json(name):
    f = RES / "kv-router" / f"{name}.json"
    return json.load(open(f)) if f.exists() else None


# ---------------------------------------------------------------- formatting
def f(x, d=1):
    if x is None:
        return "-"
    if isinstance(x, float) and abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{d}f}"


def pct(v):
    s = f"{v:+.0f}%"
    return "0%" if s in ("+0%", "-0%") else s


def inflight(p):
    if p.get("eff_c") is None:
        return "-"
    flag = " (!)" if (p.get("loop_fill") or 1) < 0.9 and not p["n_err"] else ""
    return f"{p['eff_c']:.0f}{flag}"


def md_table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


TABLES = {}


def table(name, text):
    TABLES[name] = text.rstrip() + "\n"
    (TAB / f"{name}.md").write_text(TABLES[name])


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"figures/{name}.png")


def style(ax, title, ylabel, logx=True, cs=(1, 4, 16, 64)):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    if logx:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cs)
        ax.set_xticklabels([str(c) for c in cs])
        ax.set_xlabel("concurrency (requests in flight)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)


def annotate(ax, xs, ys, color, below=False, digits=None):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        d = digits if digits is not None else (1 if y < 100 else 0)
        ax.annotate(f(y, d), (x, y), textcoords="offset points", xytext=(0, -13) if below else (0, 7),
                    ha="center", fontsize=7, color=color)


# ---------------------------------------------------------------- 0. first principles
def step_bytes(batch, ctx):
    return W_BYTES + batch * ctx * KV_TOK


def fig_first_principles():
    # Effective bandwidth: one parameter, fitted at batch 1 (decode-heavy c1). Everything else is predicted from it.
    c1 = point("h100-agg", "decode-heavy", 1)
    bw_eff = step_bytes(1, 768) / (c1["itl"] / 1000.0)          # bytes/s
    mbu = bw_eff / (PEAK_TBS * 1e12)

    # ---- roofline
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.logspace(-0.3, 4.3, 300)
    ax.plot(x, np.minimum(PEAK_TFLOPS, PEAK_TBS * x), color="black", lw=2)
    ax.plot(x, np.minimum(PEAK_TFLOPS, PEAK_TBS * x) * mbu, color="black", lw=1, ls=":")
    ax.axvline(RIDGE, color="grey", ls="--", lw=1)
    ax.text(RIDGE * 1.08, 2.0, f"ridge ~{RIDGE:.0f} FLOP/byte", fontsize=8, color="grey")
    ax.text(40, 22, f"dotted: {mbu * 100:.1f}% of peak bandwidth,\nwhat decode actually reached", fontsize=7, color="black")

    def decode_pt(b, ctx):
        flops = b * (F_TOK + ATTN_PER_CTX * ctx)
        return flops / step_bytes(b, ctx)

    def prefill_pt(t):
        flops = t * (2 * 6.95e9 + ATTN_PER_CTX * t / 2)            # causal attention: average context t/2
        return flops / (W_BYTES + t * KV_TOK)

    # (label, intensity, colour, label offset in points)
    pts = [("decode, batch 1 (any context)", decode_pt(1, 768), PAL[0], (8, -4)),
           ("decode, batch 16, context 768", decode_pt(16, 768), PAL[0], (-170, 8)),
           ("decode, batch 64, context 7,000", decode_pt(64, 7000), PAL[0], (-180, -14)),
           ("decode, batch 64, context 3,000", decode_pt(64, 3000), PAL[0], (10, -14)),
           ("decode, batch 64, context 768", decode_pt(64, 768), PAL[0], (-175, 6)),
           ("prefill chunk, 512 tokens", prefill_pt(512), PAL[2], (-30, -30)),
           ("one 3,000-token prompt", prefill_pt(3000), PAL[2], (-50, 10)),
           ("prefill chunk, 8,192 tokens", prefill_pt(8192), PAL[2], (-70, -30))]
    for lab, ai, col, off in pts:
        y = min(PEAK_TFLOPS, PEAK_TBS * ai)
        ax.scatter([ai], [y], color=col, zorder=5, s=36)
        ax.annotate(lab, (ai, y), textcoords="offset points", xytext=off, fontsize=7.5, color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=col))
    ax.text(1.0, 300, "memory-bound:\ntime = bytes / bandwidth", fontsize=9, color=PAL[0])
    ax.text(700, 250, "compute-bound:\ntime = FLOPs / FLOPS", fontsize=9, color=PAL[2])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.5, 2e4); ax.set_ylim(1, 2000)
    ax.set_xlabel("arithmetic intensity of one engine step (FLOP per byte read from HBM)")
    ax.set_ylabel("attainable TFLOPS")
    ax.set_title("H100 SXM roofline for Qwen3-8B BF16: where each kind of engine step sits (analytic)", fontsize=10, loc="left")
    ax.grid(alpha=0.3, which="both")
    save(fig, "roofline-h100")

    # ---- decode step-time model vs measured
    fig, ax = plt.subplots(figsize=(10, 5))
    b = np.arange(1, 129)
    for ctx, col in ((768, PAL[0]), (3064, PAL[1]), (7000, PAL[2])):
        t = step_bytes(b, ctx) / bw_eff * 1000
        ax.plot(b, t, color=col, lw=1.6, label=f"model, average context {ctx:,} tokens")
    meas = series("h100-agg", "decode-heavy", (1, 4, 16, 64))
    ax.scatter([p["eff_c"] for p in meas], [p["itl"] for p in meas], color=PAL[0], zorder=5, s=40,
               label="measured TPOT, decode-heavy (context ~768)")
    for p in meas:
        if p["c"] != 4:
            ax.annotate(f"{p['itl']:.2f} ms", (p["eff_c"], p["itl"]), textcoords="offset points", xytext=(4, -13), fontsize=7)
    ph = point("h100-agg", "prefill-heavy", 64)
    # Requests decoding on average = requests in flight x share of each request's life spent after its first token
    b_ph = ph["eff_c"] * (1 - ph["ttft_avg"] / 1000.0 / ph["e2e_avg"])
    ax.scatter([b_ph], [ph["gap_p50"]], color=PAL[1], marker="s", zorder=5, s=40,
               label=f"measured median token gap, prefill-heavy c64 (~{b_ph:.0f} decoding, context ~3,064)")
    ax.annotate(f"{ph['gap_p50']:.1f} ms", (b_ph, ph["gap_p50"]), textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.axhline(W_BYTES / bw_eff * 1000, color="grey", ls=":", lw=1)
    ax.text(70, W_BYTES / bw_eff * 1000 - 1.6, "weights alone (15.14 GB per step)", fontsize=7, color="grey")
    ax.set_xlabel("requests decoding in the same step (batch)")
    ax.set_ylabel("decode step time = TPOT (ms)")
    ax.set_title(f"Decode step time = (weight bytes + batch x context x 147,456 B) / {bw_eff / 1e12:.2f} TB/s "
                 "(one constant, fitted at batch 1)", fontsize=10, loc="left")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="upper left"); ax.set_ylim(0, 45)
    save(fig, "decode-step-model")

    rows = []
    for p in meas:
        pred = step_bytes(p["eff_c"], 768) / bw_eff * 1000
        rows.append([f"decode-heavy c{p['c']}", f(p["eff_c"], 0), "768", f(pred, 2), f(p["itl"], 2),
                     pct(100 * (p["itl"] / pred - 1))])
    pred = step_bytes(b_ph, 3064) / bw_eff * 1000
    rows.append(["prefill-heavy c64 (median gap)", f"~{b_ph:.0f}", "~3,064", f(pred, 2), f(ph["gap_p50"], 2),
                 pct(100 * (ph["gap_p50"] / pred - 1))])
    table("decode-model", md_table(["point", "batch", "avg context", "predicted ms", "measured ms", "error"], rows))
    return bw_eff


def fig_interference():
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, shape in zip(axs, ("decode-heavy", "prefill-heavy")):
        S = series("h100-agg", shape, (1, 4, 16, 64))
        xs = [p["c"] for p in S]
        for m, lab, col, ls in (("gap_p50", "median gap between tokens", PAL[0], "-"),
                                ("itl", "mean gap (= TPOT)", PAL[1], "-"),
                                ("gap_p90", "90th percentile gap", PAL[2], "--"),
                                ("gap_p99", "99th percentile gap", PAL[3], ":")):
            ys = [p[m] for p in S]
            ax.plot(xs, ys, marker="o", color=col, ls=ls, lw=2, label=lab)
            if shape == "prefill-heavy" and m != "gap_p99":
                annotate(ax, xs[2:], ys[2:], col, below=(m == "gap_p50"))
        ax.set_yscale("log")
        isl, osl = SHAPES[shape]
        style(ax, f"{shape} ({isl} in / {osl} out), one H100, chunk 8192", "ms between consecutive output tokens (log)")
    axs[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Prefill-decode interference: decode-only steps stay near the bandwidth floor; steps that also carry a "
                 "prompt chunk take ~200 ms", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "interference-h100")
    rows = []
    for shape in ("decode-heavy", "prefill-heavy"):
        for p in series("h100-agg", shape, (1, 4, 16, 64)):
            share = 100 * (p["itl"] - p["gap_p50"]) / p["itl"] if p["itl"] else None
            rows.append([shape, p["c"], f(p["gap_p50"], 2), f(p["itl"], 2), f(p["gap_p90"], 1), f(p["gap_p99"], 1),
                         f(max(share, 0), 0) + "%"])
    table("interference", md_table(["shape", "c", "median gap ms", "mean gap (TPOT) ms", "p90 gap ms", "p99 gap ms",
                                    "share of TPOT above the median"], rows))


# ---------------------------------------------------------------- 1. aggregated serving, H100
def fig_agg():
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
    for row, shape in enumerate(("prefill-heavy", "decode-heavy")):
        S = series("h100-agg", shape, (1, 4, 16, 64))
        xs = [p["c"] for p in S]
        for col, (m, yl, title, logy) in enumerate((("ttft", "ms (log)", "TTFT p50 and p99", True),
                                                   ("itl", "ms", "TPOT (mean gap between tokens)", False),
                                                   ("out_tps_gpu", "tokens/s", "output tokens/s per GPU and per user", False))):
            ax = axs[row][col]
            if m == "ttft":
                for k, lab, c_ in (("ttft_p50", "p50", PAL[0]), ("ttft_p99", "p99", PAL[2])):
                    ys = [p[k] for p in S]
                    ax.plot(xs, ys, "o-", color=c_, lw=2, label=lab)
                    annotate(ax, xs, ys, c_, below=(k == "ttft_p50"))
                ax.legend(fontsize=8)
            elif m == "out_tps_gpu":
                ys = [p[m] for p in S]
                ax.plot(xs, ys, "o-", color=PAL[0], lw=2, label="per GPU (throughput)")
                annotate(ax, xs, ys, PAL[0])
                ax2 = ax.twinx()
                yu = [p["tps_user"] for p in S]
                ax2.plot(xs, yu, "s--", color=PAL[1], lw=1.5, label="per user (interactivity)")
                annotate(ax2, xs, yu, PAL[1], below=True)
                ax2.set_ylabel("tokens/s per user", fontsize=9, color=PAL[1])
                ax2.tick_params(labelsize=8)
                ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="center left")
            else:
                ys = [p[m] for p in S]
                ax.plot(xs, ys, "o-", color=PAL[0], lw=2)
                annotate(ax, xs, ys, PAL[0], digits=2)
                ax.set_ylim(bottom=0, top=max(ys) * 1.15)
            if logy:
                ax.set_yscale("log")
            isl, osl = SHAPES[shape]
            style(ax, f"{shape} ({isl}/{osl}): {title}", yl)
    fig.suptitle("Aggregated Qwen3-8B BF16 on one H100: every metric against concurrency", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "agg-h100")

    for shape in ("prefill-heavy", "decode-heavy"):
        rows = []
        for p in series("h100-agg", shape, (1, 4, 16, 64)):
            rows.append([p["c"], f(p["req_s"], 2), f(p["ttft_p50"], 0), f(p["ttft_p99"], 0), f(p["itl"], 2),
                         f(p["out_tps_gpu"], 0), f(p["in_tps_gpu"], 0), f(p["tps_user"], 1), f(p.get("power_w"), 0),
                         f(p.get("tok_per_j"), 2), inflight(p), f"{p['n_ok']}/{p['n_err']}"])
        table(f"agg-{shape}", md_table(["c", "req/s", "TTFT p50 ms", "TTFT p99 ms", "TPOT ms", "out tok/s per GPU",
                                        "in tok/s per GPU", "tok/s per user", "avg power W", "output tok per joule",
                                        "avg in flight", "ok/err"], rows))

    # A100 comparison: one chart, ratio against the hardware ratios
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (shape, m, lab) in enumerate([("prefill-heavy", "req_s", "prefill-heavy: requests/s"),
                                         ("decode-heavy", "out_tps_gpu", "decode-heavy: output tok/s per GPU")]):
        A, H = series("a100-agg", shape, (1, 4, 16, 64)), series("h100-agg", shape, (1, 4, 16, 64))
        ys = [h[m] / a[m] for a, h in zip(A, H)]
        ax.plot([p["c"] for p in A], ys, "o-", lw=2, color=PAL[i], label=lab + " (vLLM defaults)")
        annotate(ax, [p["c"] for p in A], ys, PAL[i], digits=2)
    A8, H = series("a100-mbt8192", "prefill-heavy", (1, 4, 16)), series("h100-agg", "prefill-heavy", (1, 4, 16))
    ys = [h["req_s"] / a["req_s"] for a, h in zip(A8, H)]
    ax.plot([p["c"] for p in A8], ys, "s--", color=PAL[0], alpha=0.6, lw=1.5,
            label="prefill-heavy: requests/s, both at chunk 8192 (like for like)")
    ax.axhline(989 / 312, ls="--", color="black", lw=1)
    ax.text(1, 989 / 312 + 0.04, "compute ratio 3.17x (989 / 312 TFLOPS)", fontsize=8)
    ax.axhline(3350 / 1555, ls=":", color="black", lw=1)
    ax.text(1, 3350 / 1555 + 0.04, "bandwidth ratio 2.15x (3,350 / 1,555 GB/s)", fontsize=8)
    style(ax, "H100 over A100, same model, manifest and harness", "H100 / A100")
    ax.set_ylim(1.5, 3.6)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "scaling-ratio")
    rows = []
    for shape, m in (("prefill-heavy", "req_s"), ("decode-heavy", "out_tps_gpu")):
        A, H = series("a100-agg", shape, (1, 4, 16, 64)), series("h100-agg", shape, (1, 4, 16, 64))
        for a, h in zip(A, H):
            rows.append([shape, a["c"], f(a[m], 1), f(h[m], 1), f(h[m] / a[m], 2) + "x", f(a["itl"], 2), f(h["itl"], 2)])
    table("a100-vs-h100", md_table(["shape", "c", "A100 (req/s or out tok/s)", "H100", "ratio", "A100 TPOT ms",
                                    "H100 TPOT ms"], rows))

    # throughput vs interactivity frontier
    fig, ax = plt.subplots(figsize=(9, 6))
    for key, lab, color, ls in (("h100-agg", "H100 BF16", PAL[0], "-"), ("fp8", "H100 FP8 weights (c64 client-limited)", PAL[1], "-"),
                                ("best", "H100 combined config", PAL[2], "-"), ("a100-agg", "A100 BF16 (reference)", COL["A100"], "--")):
        S = series(key, "decode-heavy")
        ax.plot([p["tps_user"] for p in S], [p["out_tps_gpu"] for p in S], marker="o", ls=ls, lw=2, color=color, label=lab)
        for p in S:
            ax.annotate(f"c{p['c']}", (p["tps_user"], p["out_tps_gpu"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("output tokens/s per user (interactivity)  ->  better", fontsize=9)
    ax.set_ylabel("output tokens/s per GPU (throughput)  ->  better", fontsize=9)
    ax.set_title("Throughput vs interactivity, decode-heavy (256 in / 1024 out). Up and to the right wins.", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "frontier-decode")


# ---------------------------------------------------------------- 2. KV-aware routing
NAMES = {"rr": "round-robin", "kv": "kv (defaults)", "kv-approx": "kv, no events (TTL)",
         "kv-pls4": "kv + load scale 4", "kv-pls10": "kv + load scale 10",
         "approx-ttl-pls10": "no events (TTL) + load scale 10", "approx-lru-pls10": "no events (LRU) + load scale 10"}


def fig_kv():
    a = {arm: kv_summary("a100", "kvr-" + arm) for arm in KV_ARMS_A100}
    h = {arm: kv_summary("h100", "kvr-" + arm) for arm in KV_ARMS_H100}
    tr = {arm: (res_json(arm) or {}).get("trace") for arm in KV_ARMS_A100 + KV_ARMS_H100}

    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(KV_ARMS_A100))
    act = [a[k]["engine_hit_rate_total"] for k in KV_ARMS_A100]
    pred = [a[k]["router_predicted_hit_rate"] or 0 for k in KV_ARMS_A100]
    b1 = ax.bar(x - 0.2, act, 0.4, color=PAL[0], label="actual (vLLM prefix-cache counters)")
    b2 = ax.bar(x + 0.2, pred, 0.4, color="#bbbbbb", label="router's own prediction")
    ax.bar_label(b1, fmt="%.3f", fontsize=7)
    ax.bar_label(b2, labels=[f"{v:.3f}" if v else "n/a" for v in pred], fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[k] for k in KV_ARMS_A100], fontsize=8, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("prefix-cache hit rate (share of prompt tokens)")
    ax.set_title("Seven router configurations on the same prefix workload (40 prefixes x 2,048 tokens, 800 requests, "
                 "c=16, 2 workers)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "kv-router-arms")

    common = [("rr", "h100-rr"), ("kv", "h100-kv"), ("kv-pls10", "h100-kv-pls10")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(3)
    for ax, (m, yl, title) in zip(axs, [("hit", "hit rate", "Prefix-cache hit rate"),
                                         ("p50", "ms", "TTFT p50 (client)"), ("p90", "ms", "TTFT p90 (client)")]):
        for i, (lab, src, arms) in enumerate((("A100, 2 workers time-sliced on 1 GPU", a, [c[0] for c in common]),
                                              ("H100, 1 GPU per worker", h, [c[1] for c in common]))):
            vals = [src[k]["engine_hit_rate_total"] if m == "hit" else src[k]["ttft_ms"][m] for k in arms]
            bars = ax.bar(x + (i - 0.5) * 0.4, vals, 0.4, color=COL["A100" if i == 0 else "H100"], label=lab)
            ax.bar_label(bars, fmt="%.3f" if m == "hit" else "%.0f", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(["round-robin", "kv (defaults)", "kv + load scale 10"], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(yl)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=7, loc="upper left")
    axs[0].set_ylim(0, 1.15)
    fig.suptitle("Same routing A/B, same 7 GiB KV per worker: the hit rate is a property of the router, the TTFT gain "
                 "depends on how expensive a miss is", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "kv-two-gpus")

    rows, labs, hit, miss = [], [], [], []
    for k in KV_ARMS_A100 + KV_ARMS_H100:
        t = tr.get(k)
        if not t or "ttft_cache_hit" not in t:
            continue
        node = "H100" if k.startswith("h100") else "A100"
        labs.append(f"{node}\n{NAMES.get(k.replace('h100-', ''), k)}")
        hit.append(t["ttft_cache_hit"]["ttft_p50"])
        miss.append(t["ttft_cache_miss"]["ttft_p50"])
        rows.append([node, NAMES.get(k.replace("h100-", ""), k), f(t.get("prefix_affinity"), 2),
                     t["ttft_cache_hit"]["n"], f(t["ttft_cache_hit"]["ttft_p50"], 1),
                     t["ttft_cache_miss"]["n"], f(t["ttft_cache_miss"]["ttft_p50"], 1),
                     f(t["ttft_cache_miss"]["ttft_p90"], 1)])
    table("kv-hit-miss", md_table(["GPU", "arm", "prefix affinity", "hit n", "hit TTFT p50 ms", "miss n",
                                   "miss TTFT p50 ms", "miss TTFT p90 ms"], rows))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(labs))
    b1 = ax.bar(x - 0.2, hit, 0.4, color=PAL[0], label="requests labelled hit")
    b2 = ax.bar(x + 0.2, miss, 0.4, color=PAL[2], label="requests labelled miss")
    ax.bar_label(b1, fmt="%.0f", fontsize=7)
    ax.bar_label(b2, fmt="%.0f", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel("TTFT p50, server side (ms)")
    ax.set_title("What a cache miss costs: TTFT of hits vs misses (Dynamo request trace; in kv arms the label is the "
                 "router's prediction)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "kv-hit-vs-miss-ttft")

    rows = []
    for node, arms, src in (("A100", KV_ARMS_A100, a), ("H100", KV_ARMS_H100, h)):
        for k in arms:
            s, t = src[k], tr.get(k) or {}
            rows.append([node, NAMES[k.replace("h100-", "")], f(s["engine_hit_rate_total"], 3),
                         f(s["router_predicted_hit_rate"], 3), f(t.get("prefix_affinity"), 2), f(s["ttft_ms"]["p50"], 1),
                         f(s["ttft_ms"]["p90"], 1), f(s["itl_ms_avg"], 2),
                         "/".join(str(v) for v in s["requests_by_worker"].values())])
    table("kv-arms", md_table(["GPU", "arm", "actual hit rate", "router predicted", "prefix affinity",
                               "TTFT p50 ms", "TTFT p90 ms", "ITL ms", "requests per worker"], rows))

    arms = ["rf-default", "rf-base", "rf-queue", "rf-decay", "rf-temp"]
    lab = {"rf-default": "kv defaults", "rf-base": "+ load scale 10", "rf-queue": "+ load scale 10\n+ queue\nthreshold 0.5",
           "rf-decay": "+ load scale 10\n+ credit\ndecay 1.0", "rf-temp": "+ load scale 10\n+ temperature\n0.5"}
    R = {k: res_json(k) for k in arms}
    P = {k: json.load(open(RES / "kv-router" / f"{k}.prom.json")) for k in arms}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(arms))
    vals = [R[k]["summary"]["engine_hit_rate_total"] for k in arms]
    axs[0].bar_label(axs[0].bar(x, vals, color=PAL[0]), fmt="%.3f", fontsize=7)
    axs[0].set_title("Prefix-cache hit rate", fontsize=10, loc="left")
    p50 = [R[k]["summary"]["ttft_ms"]["p50"] for k in arms]
    p90 = [R[k]["summary"]["ttft_ms"]["p90"] for k in arms]
    axs[1].bar_label(axs[1].bar(x - 0.2, p50, 0.4, color=PAL[1], label="p50"), fmt="%.0f", fontsize=7)
    axs[1].bar_label(axs[1].bar(x + 0.2, p90, 0.4, color=PAL[2], label="p90"), fmt="%.0f", fontsize=7)
    axs[1].set_title("TTFT (ms)", fontsize=10, loc="left")
    axs[1].legend(fontsize=8)
    ov = [P[k]["routing_overhead_ms avg"] for k in arms]
    axs[2].bar_label(axs[2].bar(x, ov, color=PAL[3]), fmt="%.2f", fontsize=7)
    axs[2].set_yscale("log")
    axs[2].set_title("Time a request spends in the router (ms, log)", fontsize=10, loc="left")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([lab[k] for k in arms], fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Router features under overload: the same prefix workload at c=64 on 2x H100 (KV demand exceeds the "
                 "fleet's cache)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "router-features")
    rows = [[lab[k].replace("\n", " "), R[k]["router_args"].replace("kv ", "").strip() or "-",
             f(R[k]["summary"]["engine_hit_rate_total"], 3), f((R[k].get("trace") or {}).get("prefix_affinity"), 2),
             f(R[k]["summary"]["ttft_ms"]["p50"], 1), f(R[k]["summary"]["ttft_ms"]["p90"], 1),
             f(R[k]["summary"]["itl_ms_avg"], 2), f(P[k]["routing_overhead_ms avg"], 2),
             f(P[k]["active_prefill_tokens max per worker"], 0)] for k in arms]
    table("router-features", md_table(["arm", "extra flags", "hit rate", "prefix affinity", "TTFT p50 ms",
                                       "TTFT p90 ms", "ITL ms", "router time ms", "max active prefill tokens/worker"], rows))


# ---------------------------------------------------------------- 3. P/D vs aggregated
def fig_pd():
    keys = ["agg2", "pd-tcp", "pd-nvlink", "pd-nvlink-bs64"]
    rows = []
    for shape in ("prefill-heavy", "decode-heavy", "long"):
        for c in (16, 64):
            for k in keys:
                p = point(k, shape, c)
                if p:
                    rows.append([shape, c, RUNS[k][3], f(p["req_s"], 2), f(p["ttft_p50"], 0), f(p["ttft_p99"], 0),
                                 f(p["itl"], 2), f(p["out_tps_gpu"], 0), f(p["tps_user"], 1), inflight(p), f"{p['n_ok']}/{p['n_err']}"])
    table("pd-vs-agg", md_table(["shape", "c", "deployment (2 GPUs each)", "req/s", "TTFT p50 ms", "TTFT p99 ms",
                                 "TPOT ms", "out tok/s per GPU", "tok/s per user", "avg in flight", "ok/err"], rows))
    fig, axs = plt.subplots(3, 3, figsize=(15, 11))
    for row, shape in enumerate(("prefill-heavy", "decode-heavy", "long")):
        for col, (m, yl, logy) in enumerate((("req_s", "requests/s", False), ("ttft_p50", "TTFT p50 ms (log)", True),
                                             ("itl", "TPOT ms", False))):
            ax = axs[row][col]
            x = np.arange(2)
            for i, k in enumerate(keys):
                vals, errs = [], []
                for c in (16, 64):
                    p = point(k, shape, c)
                    vals.append(p[m] if p else 0)
                    errs.append(p["n_err"] if p else 0)
                bars = ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=PAL[i], label=RUNS[k][3])
                ax.bar_label(bars, labels=[("ERR " if e else "") + (f(v, 1) if v < 100 else f(v, 0)) if v else "n/a"
                                           for v, e in zip(vals, errs)], fontsize=6, rotation=90, padding=2)
            if logy:
                ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels(["c=16", "c=64"])
            isl, osl = SHAPES[shape]
            ax.set_title(f"{shape} ({isl}/{osl}): {yl}", fontsize=9, loc="left")
            ax.grid(axis="y", alpha=0.3)
            ax.margins(y=0.25)
    axs[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Same 2 H100s: two aggregated workers vs 1 prefill + 1 decode, by KV transport. "
                 "'ERR' = the run had failed requests.", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "pd-vs-agg")

    pat = re.compile(r"Avg xfer time \(ms\)=([0-9.]+).*?Avg post time \(ms\)=([0-9.]+).*?Avg MB per transfer=([0-9.]+), "
                     r"Throughput \(MB/s\)=([0-9.]+), Avg number of descriptors=([0-9.]+)")
    groups = {"TCP (1 GPU per pod)": [], "TCP + hostIPC": [], "MIG slices (TCP)": [], "NVLink, idle": [],
              "NVLink, c=64 load, block 16": [], "NVLink, c=64 load, block 64": []}
    for ln in (LOGS / "pd-nixl-xfer-metrics.log").read_text().splitlines():
        m = pat.search(ln)
        if not m:
            continue
        v = tuple(float(x) for x in m.groups())
        t = re.match(r"T\+(\d+):(\d+)", ln)
        if not t:
            groups["NVLink, c=64 load, block 16"].append(v)       # lines captured without timestamps
            continue
        minute = int(t.group(1))                                 # variant order is documented in the log header
        groups["TCP (1 GPU per pod)" if minute < 2 else "TCP + hostIPC" if minute < 10 else
               "NVLink, idle" if minute < 60 else "MIG slices (TCP)"].append(v)
    with gzip.open(LOGS / "pd-bs64-decode-stall.log.gz", "rt") as fh:
        for ln in fh:
            m = pat.search(ln)
            if m and "KV Transfer metrics" in ln:
                groups["NVLink, c=64 load, block 64"].append(tuple(float(x) for x in m.groups()))
    rows, labels, xfer, post, bw = [], [], [], [], []
    for g, vs in groups.items():
        if not vs:
            continue
        mean = np.array(vs).mean(axis=0)
        labels.append(g)
        gbs = mean[3] * 2**20 / 1e9          # vLLM logs MiB and MiB/s; report decimal GB/s
        xfer.append(mean[0]); post.append(mean[1]); bw.append(gbs)
        rows.append([g, len(vs), f(mean[2], 1), f(mean[4], 0) if mean[4] >= 100 else f(mean[4], 1),
                     f(mean[0], 1), f(mean[1], 2), f(100 * mean[1] / mean[0], 1) + "%", f(gbs, 2)])
    table("nixl-transfers", md_table(["variant", "10 s log intervals", "MiB per transfer (mean)", "descriptors per transfer",
                                      "transfer ms", "post ms", "post share", "GB/s (decimal)"], rows))
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    y = np.arange(len(labels))
    axs[0].barh(y, bw, color=[PAL[2] if "TCP" in l else PAL[0] for l in labels])
    for i, v in enumerate(bw):
        axs[0].text(v * 1.1, i, f"{v:.2f} GB/s", va="center", fontsize=8)
    axs[0].set_xscale("log")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(labels, fontsize=8)
    axs[0].set_xlabel("effective KV transfer bandwidth, GB/s (log)")
    axs[0].set_title("How fast does the KV cache move from prefill to decode?", fontsize=10, loc="left")
    axs[0].invert_yaxis()
    axs[1].barh(y, post, color="#555555", label="posting (build + submit the descriptor list)")
    axs[1].barh(y, np.array(xfer) - np.array(post), left=post, color="#9ecae1", label="rest of the transfer")
    for i, (p, x_) in enumerate(zip(post, xfer)):
        axs[1].text(x_ * 1.1, i, f"{x_:.1f} ms ({100 * p / x_:.1f}% posting)", va="center", fontsize=8)
    axs[1].set_xscale("log")
    axs[1].set_yticks(y)
    axs[1].set_yticklabels([])
    axs[1].invert_yaxis()
    axs[1].set_xlabel("time per transfer, ms (log)")
    axs[1].set_title("Where the transfer time goes", fontsize=10, loc="left")
    axs[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    save(fig, "nixl-transfers")


# ---------------------------------------------------------------- 4. engine tuning
def fig_tuning():
    variants = ["mbt512", "mbt8k", "fp8", "fp8-fused", "kvfp8", "seqs128", "best"]
    cells = [(s, c) for s in ("prefill-heavy", "decode-heavy", "long") for c in (16, 64)]
    metrics = [("req_s", "req/s", 1), ("itl", "TPOT", -1), ("ttft_p50", "TTFT p50", -1)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (m, name, sign) in zip(axs, metrics):
        M = np.full((len(variants), len(cells)), np.nan)
        F = np.zeros_like(M, dtype=bool)
        for i, v in enumerate(variants):
            for j, (s, c) in enumerate(cells):
                b, p = point("base", s, c), point(v, s, c)
                if b and p and b[m] and p[m]:
                    M[i, j] = 100 * (p[m] / b[m] - 1)
                    F[i, j] = (p.get("loop_fill") or 1) < 0.9
        im = ax.imshow(np.clip(sign * M, -60, 60), cmap="RdYlGn", vmin=-60, vmax=60, aspect="auto")
        for i in range(len(variants)):
            for j in range(len(cells)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.0f}%" + (" (!)" if F[i, j] else ""), ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{s.split('-')[0]}\nc{c}" for s, c in cells], fontsize=7)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([RUNS[v][3] for v in variants] if ax is axs[0] else [], fontsize=7)
        ax.set_title(f"{name} change vs BF16 baseline", fontsize=10, loc="left")
    fig.colorbar(im, ax=axs, shrink=0.8, label="green = better, red = worse (clipped at 60%)")
    fig.suptitle("One H100, Qwen3-8B: each engine setting against the paired BF16 baseline run. "
                 "(!) = load generator kept < 90% of the requested concurrency, so req/s is understated",
                 fontsize=11, x=0.01, ha="left")
    save(fig, "tuning-heatmap")

    def group_table(name, keys, shapes=("prefill-heavy", "decode-heavy", "long"), cs=(16, 64, 128)):
        rows = []
        for s in shapes:
            for c in cs:
                for k in keys:
                    p = point(k, s, c)
                    if p:
                        rows.append([s, c, RUNS[k][3], f(p["req_s"], 2), f(p["ttft_p50"], 0), f(p["ttft_p99"], 0),
                                     f(p["itl"], 2), f(p["out_tps_gpu"], 0), f(p["tps_user"], 1), inflight(p), f"{p['n_ok']}/{p['n_err']}"])
        table(name, md_table(["shape", "c", "config", "req/s", "TTFT p50 ms", "TTFT p99 ms", "TPOT ms",
                              "out tok/s per GPU", "tok/s per user", "avg in flight", "ok/err"], rows))
    group_table("tuning-chunk", ["mbt512", "base", "mbt8k"])
    group_table("tuning-fp8", ["base", "fp8", "fp8-fused", "kvfp8"])
    group_table("tuning-batch", ["base", "seqs128", "best"])

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (m, yl) in zip(axs, (("req_s", "requests/s"), ("ttft_p50", "TTFT p50 ms (log)"), ("itl", "TPOT ms"))):
        for i, (k, lab) in enumerate((("mbt512", "512"), ("base", "8192 (H100 default)"), ("mbt8k", "8192 again (repeat)"))):
            S = series(k, "prefill-heavy", (16, 64))
            ax.plot([p["c"] for p in S], [p[m] for p in S], "o-", lw=2, color=PAL[i], label=f"budget {lab}")
            annotate(ax, [p["c"] for p in S], [p[m] for p in S], PAL[i], below=(i % 2 == 1))
        if m == "ttft_p50":
            ax.set_yscale("log")
        style(ax, yl, yl, cs=(16, 64))
        ax.legend(fontsize=8)
    fig.suptitle("Chunked-prefill token budget (--max-num-batched-tokens), prefill-heavy 3000/128, one H100",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "tuning-chunk-budget")

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (k, lab) in enumerate((("h100-agg", "no speculation"), ("spec-ngram", "n-gram (prompt lookup) k=4"),
                                  ("spec-draft", "Qwen3-0.6B draft model k=4"))):
        S = series(k, "decode-heavy", (1, 4, 16, 64))
        for ax, m in zip(axs, ("itl", "out_tps_gpu")):
            ax.plot([p["c"] for p in S], [p[m] for p in S], "o-", lw=2, color=PAL[i], label=lab)
            annotate(ax, [p["c"] for p in S], [p[m] for p in S], PAL[i], below=(i % 2 == 1), digits=None)
    style(axs[0], "TPOT (ms, lower is better)", "ms")
    style(axs[1], "Output tokens/s per GPU", "tok/s")
    axs[0].legend(fontsize=8)
    fig.suptitle("Speculative decoding, decode-heavy 256/1024 (synthetic text makes acceptance optimistic)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "tuning-spec-decode")
    rows = []
    for k in ("h100-agg", "spec-ngram", "spec-draft"):
        for p in series(k, "decode-heavy", (1, 4, 16, 64)):
            rows.append(["no speculation" if k == "h100-agg" else RUNS[k][3], p["c"], f(p["itl"], 2),
                         f(p["out_tps_gpu"], 0), f(p["tps_user"], 1), f(p["ttft_p50"], 0), inflight(p), f"{p['n_ok']}/{p['n_err']}"])
    table("tuning-spec", md_table(["config", "c", "TPOT ms", "out tok/s per GPU", "tok/s per user", "TTFT p50 ms",
                                   "avg in flight", "ok/err"], rows))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (s, m, yl) in zip(axs, (("decode-heavy", "out_tps_gpu", "decode-heavy: out tok/s per GPU"),
                                    ("prefill-heavy", "req_s", "prefill-heavy: requests/s"),
                                    ("long", "req_s", "long 6000/1024: requests/s"))):
        for i, k in enumerate(("base", "seqs128", "best")):
            S = series(k, s, (16, 64, 128))
            ax.plot([p["c"] for p in S], [p[m] for p in S], "o-", lw=2, color=PAL[i], label=RUNS[k][3])
            annotate(ax, [p["c"] for p in S], [p[m] for p in S], PAL[i], below=(i % 2 == 1),
                     digits=2 if m == "req_s" else 0)
        style(ax, yl, yl, cs=(16, 64, 128))
    axs[0].legend(fontsize=7)
    fig.suptitle("Batch slots and the combined configuration, one H100 (the BF16 baseline caps at 64 sequences)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "tuning-combined")


# ---------------------------------------------------------------- 5. MIG + planner
PLANNER_T0_S = 216_163   # results offset (s) of the replica log's first line, T+0:00 = planner graph deployed


def fig_planner():
    t, dp, dd, rp, rd = [], [], [], [], []
    for ln in (LOGS / "mig-planner-replicas.log").read_text().splitlines()[1:]:
        m = re.match(r"T\+(\d+):(\d+) desired=\((\d+),(\d+)\) ready=\((\d+),(\d+)\)", ln)
        if not m:
            continue
        mi, s, a, b, c, d = map(int, m.groups())
        t.append(mi + s / 60)
        dp.append(a); dd.append(b); rp.append(c); rd.append(d)
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.step(t, dp, where="post", color=PAL[1], lw=2, label="prefill desired")
    ax.step(t, rp, where="post", color=PAL[1], lw=1, ls="--", label="prefill ready")
    ax.step(t, dd, where="post", color=PAL[2], lw=2, label="decode desired")
    ax.step(t, rd, where="post", color=PAL[2], lw=1, ls="--", label="decode ready")
    win = {}
    for ph in ("prefill", "decode", "mixed-p"):
        e = IDX["h100"][f"autoscale/{ph}/profile_export_aiperf.json"]
        win[ph] = ((e["start_time_offset_s"] - PLANNER_T0_S) / 60, (e["end_time_offset_s"] - PLANNER_T0_S) / 60)
    for (start, end), lab, col in ((win["prefill"], "prefill-heavy load", "#dbe9f6"),
                                   (win["decode"], "decode-heavy load", "#f6dbdb"),
                                   (win["mixed-p"], "mixed load", "#e8e8e8")):
        ax.axvspan(start, end, color=col, zorder=0)
        ax.text(start + 0.2, 3.3, lab, fontsize=8)
    ax.set_ylim(0, 3.6)
    ax.set_xlabel("minutes since the planner graph was deployed")
    ax.set_ylabel("replicas (1 MIG slice each)")
    ax.set_title("Dynamo planner on 4 MIG slices: desired vs ready replicas per pool (gap = pod cold start)",
                 fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    save(fig, "planner-replicas")

    rows = []
    for ph in ("prefill", "decode", "mixed-p", "mixed-d"):
        p = IDX["h100"].get(f"autoscale/{ph}/profile_export_aiperf.json")
        if not p:
            continue
        g = lambda k, s="avg": (p.get(k) or {}).get(s)
        err = sum(e.get("count", 0) for e in (p.get("error_summary") or []))
        rows.append([ph, f(g("request_throughput"), 2), f(g("time_to_first_token", "p50"), 0),
                     f(g("time_to_first_token", "p99"), 0), f(g("inter_token_latency"), 2),
                     f(g("output_token_throughput"), 0), f((p.get("effective_concurrency") or {}).get("avg"), 1),
                     f"{int(g('request_count') or 0)}/{err}"])
    table("planner-phases", md_table(["load phase", "req/s", "TTFT p50 ms", "TTFT p99 ms", "TPOT ms",
                                      "output tok/s (all slices)", "avg in flight", "ok/err"], rows))


# ---------------------------------------------------------------- splice tables into the docs
def splice():
    for md in sorted(DOCS.glob("*.md")) + [ROOT / "README.md"]:
        if not md.exists():
            continue
        s = md.read_text()
        new = re.sub(r"(<!-- BEGIN GENERATED: ([\w-]+) -->\n).*?(<!-- END GENERATED -->)",
                     lambda m: m.group(1) + (TABLES.get(m.group(2)) or f"(missing table {m.group(2)})\n") + m.group(3),
                     s, flags=re.S)
        if new != s:
            md.write_text(new)
            print("refreshed tables in", md.relative_to(ROOT))


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    bw = fig_first_principles()
    print(f"effective decode bandwidth fitted at batch 1: {bw / 1e12:.3f} TB/s ({100 * bw / (PEAK_TBS * 1e12):.1f}% of peak)")
    fig_interference(); fig_agg(); fig_kv(); fig_pd(); fig_tuning(); fig_planner()
    splice()
