#!/usr/bin/env python3
"""Plot desired vs ready prefill/decode replicas from bench/watch-replicas.sh output, with load phases shaded.
  ~/hfenv/bin/python profile/replica-chart.py ~/replicas.log out.png "<start>,<end>,prefill" "<start>,<end>,decode" ...
"""
import re, sys, datetime as dt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = []
for line in open(sys.argv[1]):
    m = re.match(r"(\d\d:\d\d:\d\d) desired=\((\d+),(\d+)\) ready=\((\d+),(\d+)\)", line)
    if m: rows.append((dt.datetime.strptime(m.group(1), "%H:%M:%S"), *map(int, m.groups()[1:])))
t = [r[0] for r in rows]
fig, ax = plt.subplots(figsize=(15, 4.5))
cols = {"prefill": "#fde0c5", "decode": "#d4f1d4", "mixed": "#dde6f7"}
for spec in sys.argv[3:]:
    a, b, name = spec.split(",")
    ax.axvspan(dt.datetime.strptime(a, "%H:%M:%S"), dt.datetime.strptime(b, "%H:%M:%S"), color=cols.get(name, "#eee"), alpha=0.8)
    ax.text(dt.datetime.strptime(a, "%H:%M:%S"), 3.75, f" {name}-heavy load" if name != "mixed" else " mixed load", fontsize=9)
ax.step(t, [r[1] for r in rows], where="post", color="#d9480f", lw=2, label="prefill desired")
ax.step(t, [r[3] for r in rows], where="post", color="#d9480f", lw=1, ls="--", label="prefill ready")
ax.step(t, [r[2] + 0.04 for r in rows], where="post", color="#2b8a3e", lw=2, label="decode desired")
ax.step(t, [r[4] + 0.04 for r in rows], where="post", color="#2b8a3e", lw=1, ls="--", label="decode ready")
ax.set_ylim(0, 4); ax.set_yticks([0, 1, 2, 3, 4]); ax.set_ylabel("replicas (1 replica = 1 MIG 3g.40gb slice)")
ax.set_title("Dynamo planner (disagg, optimization_target=load, GPU budget pinned 4): MIG slices moving between prefill and decode pools")
ax.legend(loc="lower right", fontsize=8); import matplotlib.dates as md; ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
plt.tight_layout(); plt.savefig(sys.argv[2], dpi=110); print("wrote", sys.argv[2])
