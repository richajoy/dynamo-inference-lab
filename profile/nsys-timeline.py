#!/usr/bin/env python3
"""Render a slice of an Nsight Systems report as a PNG timeline (the GUI is not available on a headless node).
Runs ON the node with the venv that has matplotlib:
  ~/hfenv/bin/python profile/nsys-timeline.py <report.nsys-rep> <out.png> [slice_ms=250] [start_offset_ms=auto]
Exports the report to SQLite with the host nsys, then draws two lanes for a short window:
  - NVTX: vLLM per-step ranges execute_context_<n>(<tok>)_generation_<n>(<tok>)  -> which steps are prefill/mixed/decode
  - GPU kernels, coloured by class: GEMM (cutlass/sm90/gemm/nvjet), attention (flash/fmha), other
Also prints a per-step table (duration, context tokens, generation tokens, kernel-busy %).
"""
import re, sqlite3, subprocess, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep, out = pathlib.Path(sys.argv[1]), sys.argv[2]
slice_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 250.0
db = rep.with_suffix(".sqlite")
if not db.exists():
    subprocess.run(["/opt/nvidia/nsys-current/bin/nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                    "--output", str(db), str(rep)], check=True, capture_output=True)
c = sqlite3.connect(db)
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
nvtx = []
if "NVTX_EVENTS" in tables:
    nvtx = c.execute("""select start, end, coalesce(text, (select value from StringIds where id=textId)) from NVTX_EVENTS
                        where end is not null""").fetchall()
steps = [(s, e, t) for s, e, t in nvtx if t and t.startswith("execute_context")]
kern = c.execute("""select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
                    order by k.start""").fetchall()
if not kern:
    sys.exit("no kernels in report")
# window: start at the first step that has context (prefill) tokens if any, else at the first step
def ctx(t):
    m = re.match(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", t or "")
    return (int(m.group(2)), int(m.group(4))) if m else (0, 0)
t0 = next((s for s, e, t in steps if ctx(t)[0] > 0), steps[0][0] if steps else kern[0][0])
if len(sys.argv) > 4:
    t0 = kern[0][0] + float(sys.argv[4]) * 1e6
t1 = t0 + slice_ms * 1e6
def cls(n):
    n = n.lower()
    # attention first: FlashAttention-3 kernels (FlashAttnFwdSm90) also contain "sm90"
    if any(x in n for x in ("flash", "fmha", "attention", "paged")): return "attention", "#d62728"
    if any(x in n for x in ("gemm", "cutlass", "sm90", "nvjet", "cublas", "matmul")): return "GEMM", "#1f77b4"
    return "other", "#7f7f7f"
fig, ax = plt.subplots(figsize=(18, 4.2))
for s, e, t in steps:
    if e < t0 or s > t1: continue
    cx, gx = ctx(t)
    col = "#ff7f0e" if cx > 0 else "#2ca02c"
    ax.barh(1, (e - s) / 1e6, left=(s - t0) / 1e6, color=col, edgecolor="black", linewidth=0.3)
    if (e - s) / 1e6 > slice_ms / 40:
        ax.text((s - t0) / 1e6 + 0.3, 1, f"ctx {cx}\ngen {gx}", va="center", fontsize=7)
seen = set()
for s, e, n in kern:
    if e < t0 or s > t1: continue
    k, col = cls(n)
    ax.barh(0, max((e - s) / 1e6, 0.01), left=(s - t0) / 1e6, color=col, label=None if k in seen else k)
    seen.add(k)
ax.set_yticks([0, 1]); ax.set_yticklabels(["GPU kernels", "vLLM steps (NVTX)\norange=has prefill, green=decode-only"])
ax.set_xlabel(f"ms from window start ({rep.name})"); ax.set_xlim(0, slice_ms); ax.legend(loc="upper right", fontsize=8)
ax.set_title("Nsight Systems slice: vLLM engine steps vs GPU kernels")
plt.tight_layout(); plt.savefig(out, dpi=110)
print(f"wrote {out}")
print("step_ms ctx_tokens gen_tokens kernel_busy_%")
for s, e, t in steps:
    if e < t0 or s > t1: continue
    busy = sum(min(ke, e) - max(ks, s) for ks, ke, _ in kern if ke > s and ks < e)
    cx, gx = ctx(t)
    print(f"{(e - s) / 1e6:7.2f} {cx:6d} {gx:6d} {100 * busy / max(e - s, 1):6.1f}")
