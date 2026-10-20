#!/usr/bin/env python3
"""Generate the animated explanatory diagrams in figures/ (self-contained SVG, no dependencies).

  figures/anim-request-path.svg   aggregated vs disaggregated request path
  figures/anim-interference.svg   why a prompt chunk stalls every decoding user, and what disaggregation changes
  figures/anim-kv-transfer.svg    one KV cache over NVLink vs TCP, and contiguous vs fragmented KV blocks

Each SVG animates with CSS keyframes, loops, respects prefers-reduced-motion (motion stops and a complete static
diagram remains) and prefers-color-scheme (dark palette). Numbers shown are the measured values used in the docs.
Run:  python3 figures/make_diagrams.py
"""
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

# One palette for every diagram: same meaning as the class colours in figures/diagrams/*.mmd.
STYLE = """
  .bg {{ fill: #ffffff; }}
  .ink {{ fill: #1f2328; }} .muted {{ fill: #57606a; }} .boxink {{ fill: #1f2328; }}
  .line {{ stroke: #8c959f; }}
  .client {{ fill: #eef1f4; stroke: #57606a; }}
  .dynamo {{ fill: #dbeafe; stroke: #1f6feb; }}
  .gpu {{ fill: #e8f5d0; stroke: #4c8c00; }}
  .kvbox {{ fill: #efe4ff; stroke: #8250df; }}
  .prefill {{ fill: #fde7c9; stroke: #e07b00; }}
  .decode {{ fill: #cfe8a9; stroke: #4c8c00; }}
  .wait {{ fill: #ffd8d3; stroke: #cf222e; }}
  .c-blue {{ fill: #1f6feb; }} .c-green {{ fill: #4c8c00; }} .c-orange {{ fill: #e07b00; }}
  .c-purple {{ fill: #8250df; }} .c-red {{ fill: #cf222e; }}
  .s-purple {{ stroke: #8250df; }} .s-red {{ stroke: #cf222e; }}
  text {{ font-family: {font}; }}
  .title {{ font-size: 17px; font-weight: 600; }}
  .label {{ font-size: 13px; }} .small {{ font-size: 11.5px; }} .bold {{ font-weight: 600; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ fill: #0d1117; }}
    .ink {{ fill: #e6edf3; }} .muted {{ fill: #9198a1; }}
    .line {{ stroke: #6e7681; }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; }}
    .motion {{ display: none; }}
  }}
"""


def svg(w, h, body, extra_css=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img">\n<style>{STYLE.format(font=FONT)}{extra_css}</style>\n'
            f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="10"/>\n{body}\n</svg>\n')


def box(x, y, w, h, cls, lines, title_bold=True):
    out = [f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8" stroke-width="1.6"/>']
    n = len(lines)
    for i, t in enumerate(lines):
        ty = y + h / 2 + (i - (n - 1) / 2) * 16 + 4.5
        weight = " bold" if (i == 0 and title_bold) else ""
        size = "label" if i == 0 else "small"
        out.append(f'<text class="boxink {size}{weight}" x="{x + w / 2}" y="{ty:.1f}" text-anchor="middle">{t}</text>')
    return "\n".join(out)


def arrow(x1, y1, x2, y2, cls="line", width=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="{width}"{d} '
            f'marker-end="url(#head)"/>')


MARKER = ('<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#8c959f"/></marker></defs>')


def keyframes(name, stops):
    """stops: list of (percent, css) pairs."""
    body = " ".join(f"{p:.2f}% {{ {css} }}" for p, css in stops)
    return f"\n  @keyframes {name} {{ {body} }}"


def path_anim(name, points, t0, t1, cycle_hidden=True):
    """Move a dot along straight segments between `points` from t0% to t1% of the cycle; hidden otherwise."""
    segs = [((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2) ** 0.5
            for i in range(len(points) - 1)]
    total = sum(segs) or 1
    stops = [(0, f"opacity: 0; transform: translate({points[0][0]}px, {points[0][1]}px);")]
    if t0 > 0:
        stops.append((max(t0 - 0.01, 0), f"opacity: 0; transform: translate({points[0][0]}px, {points[0][1]}px);"))
    acc = 0
    stops.append((t0, f"opacity: 1; transform: translate({points[0][0]}px, {points[0][1]}px);"))
    for i, s in enumerate(segs):
        acc += s
        p = t0 + (t1 - t0) * acc / total
        stops.append((p, f"opacity: 1; transform: translate({points[i + 1][0]}px, {points[i + 1][1]}px);"))
    stops.append((min(t1 + 0.01, 100), f"opacity: 0; transform: translate({points[-1][0]}px, {points[-1][1]}px);"))
    stops.append((100, f"opacity: 0; transform: translate({points[-1][0]}px, {points[-1][1]}px);"))
    return keyframes(name, stops)


def pulse(name, t0, t1):
    return keyframes(name, [(0, "opacity: 0;"), (max(t0 - 0.01, 0), "opacity: 0;"), (t0, "opacity: 0.85;"),
                            (t1, "opacity: 0.85;"), (min(t1 + 0.01, 100), "opacity: 0;"), (100, "opacity: 0;")])


# ---------------------------------------------------------------- 1. request path
def request_path():
    W, H, CYC = 960, 520, 7
    css, body = "", [MARKER,
                     '<text class="ink title" x="24" y="34">One request, two ways to serve it</text>',
                     '<text class="muted small" x="24" y="54">Blue: the prompt. Orange: prefill. Purple: the KV '
                     'cache moving between GPUs. Green: output tokens.</text>']
    # lane 1: aggregated
    y1 = 110
    body += ['<text class="ink label bold" x="24" y="96">Aggregated: one GPU does prefill and decode</text>',
             box(40, y1, 150, 64, "client", ["Client", "(or AIPerf)"]),
             box(300, y1, 190, 64, "dynamo", ["Dynamo frontend", "tokenize + route"]),
             box(600, y1, 300, 64, "gpu", ["vLLM worker (one GPU)", "prefill, then one decode step per token"]),
             arrow(190, y1 + 24, 298, y1 + 24), arrow(490, y1 + 24, 598, y1 + 24),
             arrow(598, y1 + 44, 492, y1 + 44), arrow(298, y1 + 44, 192, y1 + 44)]
    # lane 2: disaggregated
    y2 = 290
    body += ['<text class="ink label bold" x="24" y="268">Disaggregated: prefill and decode on separate GPUs</text>',
             box(40, y2 + 40, 150, 64, "client", ["Client", "(or AIPerf)"]),
             box(300, y2 + 40, 190, 64, "dynamo", ["Dynamo frontend", "prefill leg, then decode leg"]),
             box(600, y2, 300, 56, "gpu", ["Prefill worker (GPU 0)", "computes and stages the KV cache"]),
             box(600, y2 + 112, 300, 56, "gpu", ["Decode worker (GPU 1)", "pulls the KV, generates tokens"]),
             arrow(190, y2 + 62, 298, y2 + 62), arrow(490, y2 + 56, 598, y2 + 30),
             arrow(750, y2 + 58, 750, y2 + 110, cls="s-purple", width=2.4, dash="6 4"),
             f'<text class="c-purple small bold" x="738" y="{y2 + 74}" text-anchor="end">KV cache via NIXL / UCX</text>',
             f'<text class="muted small" x="738" y="{y2 + 88}" text-anchor="end">NVLink, or TCP if the pods</text>',
             f'<text class="muted small" x="738" y="{y2 + 102}" text-anchor="end">cannot see the peer GPU</text>',
             arrow(598, y2 + 140, 492, y2 + 86), arrow(298, y2 + 84, 192, y2 + 84)]
    # highlights (pulses on worker boxes)
    hl = [("hl-agg-pre", 600, y1, 300, 64, "prefill", 22, 38), ("hl-agg-dec", 600, y1, 300, 64, "decode", 40, 92),
          ("hl-dis-pre", 600, y2, 300, 56, "prefill", 22, 38), ("hl-dis-dec", 600, y2 + 112, 300, 56, "decode", 56, 92)]
    for name, x, y, w, h, cls, a, b in hl:
        body.append(f'<rect class="{cls} motion" x="{x}" y="{y}" width="{w}" height="{h}" rx="8" stroke-width="2.4" '
                    f'style="animation: {name} {CYC}s linear infinite; opacity: 0; fill: none; stroke-width: 5px;"/>')
        css += pulse(name, a, b)
    # moving dots
    dots = [("p-agg", [(190, y1 + 24), (300, y1 + 24), (490, y1 + 24), (600, y1 + 24)], 2, 20, "c-blue", 7),
            ("p-dis", [(190, y2 + 62), (300, y2 + 62), (490, y2 + 56), (600, y2 + 30)], 2, 20, "c-blue", 7),
            ("kv-dis", [(750, y2 + 60), (750, y2 + 110)], 40, 54, "c-purple", 8)]
    for i in range(5):  # token stream
        s = 42 + i * 10
        dots.append((f"t-agg{i}", [(600, y1 + 44), (490, y1 + 44), (300, y1 + 44), (190, y1 + 44)], s, s + 9, "c-green", 6))
        s2 = 57 + i * 7
        dots.append((f"t-dis{i}", [(600, y2 + 140), (490, y2 + 86), (300, y2 + 84), (190, y2 + 84)], s2, s2 + 8, "c-green", 6))
    for name, pts, a, b, cls, r in dots:
        body.append(f'<circle class="{cls} motion" r="{r}" cx="0" cy="0" '
                    f'style="animation: {name} {CYC}s linear infinite; opacity: 0;"/>')
        css += path_anim(name, pts, a, b)
    body.append('<text class="muted small" x="24" y="500">Dynamo decides the order of the two legs and the workers; '
                'vLLM stages and pulls the KV blocks; UCX chooses the wire.</text>')
    return svg(W, H, "\n".join(body), css)


# ---------------------------------------------------------------- 2. interference timeline
def interference():
    W, H, CYC = 960, 450, 8
    X0, PX = 170, 1.7          # timeline origin and pixels per ms
    css = ""
    body = [MARKER, '<text class="ink title" x="24" y="34">Why one long prompt stalls every decoding user</text>',
            '<text class="muted small" x="24" y="54">Measured on one H100, prefill-heavy traffic, 64 concurrent '
            'requests: most gaps between tokens are ~17 ms, one in ten is ~200 ms.</text>']

    def lane(y, label, steps, name):
        nonlocal css
        out = [f'<text class="ink label bold" x="24" y="{y + 24}">{label[0]}</text>',
               f'<text class="muted small" x="24" y="{y + 40}">{label[1]}</text>']
        t = 0
        for kind, ms in steps:
            x, w = X0 + t * PX, ms * PX
            cls = "decode" if kind == "d" else "prefill" if kind == "p" else "gpu"
            out.append(f'<rect class="{cls}" x="{x + 1:.1f}" y="{y + 8}" width="{w - 2:.1f}" height="36" rx="3" stroke-width="1"/>')
            if kind == "m":
                out.append(f'<rect class="prefill" x="{x + 1:.1f}" y="{y + 8}" width="{w - 2:.1f}" height="28" rx="3" stroke-width="1"/>')
                out.append(f'<rect class="decode" x="{x + 1:.1f}" y="{y + 36}" width="{w - 2:.1f}" height="8" rx="2" stroke-width="1"/>')
                out.append(f'<text class="boxink small bold" x="{x + w / 2:.1f}" y="{y + 27}" text-anchor="middle">'
                           f'prompt chunk (up to 8,192 tokens): ~200 ms</text>')
            t += ms
        return out, t

    agg_steps = [("d", 17)] * 6 + [("m", 200)] + [("d", 17)] * 6
    dis_steps = [("d", 17)] * 24
    y_agg, y_dis = 90, 270
    l1, tmax = lane(y_agg, ("Aggregated", "one GPU, mixed steps"), agg_steps, "agg")
    l2, _ = lane(y_dis, ("Disaggregated", "decode GPU only"), dis_steps, "dis")
    body += l1 + l2
    body.append(f'<rect class="prefill" x="{X0}" y="{y_dis + 60}" width="{200 * PX:.1f}" height="22" rx="3" stroke-width="1"/>')
    body.append(f'<text class="boxink small" x="{X0 + 100 * PX:.1f}" y="{y_dis + 75}" text-anchor="middle">the same prompt is '
                f'prefilled on the prefill GPU, in parallel</text>')
    body.append(f'<text class="muted small" x="24" y="{y_dis + 75}">prefill GPU</text>')
    # token ticks for one user (one token per step end) and the stall callout
    def ticks(y, steps, prefix):
        nonlocal css
        out, t = [], 0
        for i, (kind, ms) in enumerate(steps):
            t += ms
            x = X0 + t * PX
            a = 100 * t / (tmax * 1.15)
            name = f"{prefix}{i}"
            out.append(f'<circle class="c-green" cx="{x - 2:.1f}" cy="{y}" r="4.5" style="animation: {name} {CYC}s linear infinite;"/>')
            css += keyframes(name, [(0, "opacity: 0.18;"), (max(a - 0.01, 0), "opacity: 0.18;"), (a, "opacity: 1;"),
                                    (96, "opacity: 1;"), (100, "opacity: 0.18;")])
        return out
    body.append(f'<text class="muted small" x="24" y="{y_agg + 68}">user tokens</text>')
    body += ticks(y_agg + 64, agg_steps, "ta")
    body.append(f'<text class="muted small" x="24" y="{y_dis + 104}">user tokens</text>')
    body += ticks(y_dis + 100, dis_steps[:int(tmax // 17)], "td")
    x_a, x_b = X0 + 6 * 17 * PX, X0 + (6 * 17 + 200) * PX
    body.append(f'<rect class="wait" x="{x_a:.1f}" y="{y_agg + 54}" width="{x_b - x_a:.1f}" height="20" rx="3" '
                f'stroke-width="1" fill-opacity="0.6"/>')
    body.append(f'<text class="c-red small bold" x="{(x_a + x_b) / 2:.1f}" y="{y_agg + 68}" text-anchor="middle">'
                f'every decoding user waits</text>')
    # time axis and playhead
    ya = y_agg + 96
    body.append(f'<line class="line" x1="{X0}" y1="{ya}" x2="{X0 + tmax * PX:.1f}" y2="{ya}" stroke-width="1"/>')
    for ms in range(0, int(tmax) + 1, 100):
        body.append(f'<text class="muted small" x="{X0 + ms * PX:.1f}" y="{ya + 16}" text-anchor="middle">{ms} ms</text>')
    body.append(f'<line class="s-red motion" x1="0" y1="{y_agg}" x2="0" y2="{y_dis + 110}" stroke-width="1.5" '
                f'style="animation: playhead {CYC}s linear infinite;"/>')
    css += keyframes("playhead", [(0, f"transform: translate({X0}px, 0px); opacity: 1;"),
                                  (100 / 1.15, f"transform: translate({X0 + tmax * PX:.1f}px, 0px); opacity: 1;"),
                                  (100 / 1.15 + 0.01, "opacity: 0;"), (100, "opacity: 0;")])
    body.append('<text class="muted small" x="24" y="430">Step lengths drawn to scale. Green: decode-only step. '
                'Orange: prompt work. Each green dot is one user getting a token.</text>')
    return svg(W, H, "\n".join(body), css)


# ---------------------------------------------------------------- 3. KV transfer
def kv_transfer():
    W, H, CYC = 960, 500, 6
    css = ""
    body = [MARKER, '<text class="ink title" x="24" y="34">Moving one KV cache (846 MiB) from the prefill GPU to the decode GPU</text>',
            '<text class="muted small" x="24" y="54">Both bars run on the same clock. Measured on 2x H100: the link '
            'UCX chooses decides everything.</text>']
    X0, WBAR = 260, 600
    tcp_ms, nv_ms = 2000, 5.4
    fill_end = 100 / 1.25
    for i, (lab, sub, ms, cls) in enumerate((("CUDA IPC over NVLink", "both GPUs visible to the pods", nv_ms, "c-green"),
                                             ("TCP over the pod network", "one GPU per pod (the default)", tcp_ms, "c-red"))):
        y = 96 + i * 70
        body += [f'<text class="ink label bold" x="24" y="{y + 18}">{lab}</text>',
                 f'<text class="muted small" x="24" y="{y + 34}">{sub}</text>',
                 f'<rect class="client" x="{X0}" y="{y}" width="{WBAR}" height="30" rx="5" stroke-width="1"/>',
                 f'<rect class="{cls}" x="{X0}" y="{y}" width="{WBAR}" height="30" rx="5" '
                 f'style="transform-origin: {X0}px {y}px; animation: bar{i} {CYC}s linear infinite;"/>',
                 f'<text class="ink label bold" x="{X0 + WBAR + 14}" y="{y + 20}">{"5.4 ms" if i == 0 else "~2,000 ms"}</text>']
        end = fill_end * ms / tcp_ms
        css += keyframes(f"bar{i}", [(0, "transform: scaleX(0);"), (max(end, 0.4), "transform: scaleX(1);"),
                                     (96, "transform: scaleX(1);"), (100, "transform: scaleX(0);")])
    body.append(f'<text class="c-purple label bold" x="{X0}" y="246">About 370x apart. Over TCP a prompt-heavy service '
                f'topped out near 1 request/s.</text>')
    # descriptor panel
    yb = 300
    body.append(f'<text class="ink label bold" x="24" y="{yb - 8}">Once the link is fast: how the KV blocks are laid out decides the copy list</text>')
    cols, cell, gap = 24, 16, 3
    rnd = random.Random(7)
    picked = sorted(rnd.sample(range(cols * 3), 12))
    for p, (title, sub, sel, ndesc) in enumerate((("Idle engine", "a new request gets adjacent free blocks", set(range(27, 39)), "36 descriptors (one span per layer)"),
                                                  ("Under load", "free blocks are scattered", set(picked), "13,500 descriptors (one per block per layer)"))):
        x0 = 24 + p * 470
        body.append(f'<text class="ink small bold" x="{x0}" y="{yb + 16}">{title}</text>')
        body.append(f'<text class="muted small" x="{x0}" y="{yb + 32}">{sub}</text>')
        for r in range(3):
            for c in range(cols):
                k = r * cols + c
                x, y = x0 + c * (cell + gap), yb + 44 + r * (cell + gap)
                on = k in sel
                cls = "kvbox" if on else "client"
                anim = f' style="animation: blk{p} {CYC}s linear infinite;"' if on else ""
                body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" stroke-width="1"{anim}/>')
        body.append(f'<text class="c-purple small bold" x="{x0}" y="{yb + 124}">{ndesc}</text>')
        css += keyframes(f"blk{p}", [(0, "opacity: 0.35;"), (15 + p * 35, "opacity: 0.35;"), (20 + p * 35, "opacity: 1;"),
                                     (92, "opacity: 1;"), (100, "opacity: 0.35;")])
    body.append('<text class="muted small" x="24" y="470">Each descriptor costs ~3.8 μs of CPU work to post. 13,500 of them '
                'took more than half of a ~93 ms transfer on the fast link.</text>')
    return svg(W, H, "\n".join(body), css)


if __name__ == "__main__":
    for name, fn in (("anim-request-path", request_path), ("anim-interference", interference), ("anim-kv-transfer", kv_transfer)):
        p = OUT / f"{name}.svg"
        p.write_text(fn(), encoding='utf-8')
        print("wrote", p.relative_to(OUT.parent), f"{p.stat().st_size / 1024:.1f} KB")
