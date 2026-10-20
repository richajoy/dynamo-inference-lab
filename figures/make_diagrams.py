#!/usr/bin/env python3
"""Generate the animated explanatory diagrams in figures/ (self-contained SVG, no dependencies).

  figures/anim-request-path.svg   aggregated vs disaggregated request path
  figures/anim-interference.svg   why a prompt chunk stalls every decoding user, and what disaggregation changes
  figures/anim-kv-transfer.svg    one KV cache over NVLink vs TCP, and contiguous vs fragmented KV blocks
  figures/anim-request-journey.svg  one request: where TTFT and TPOT are spent, and the knob that moves each (README hero)
  figures/anim-stack-ownership.svg  who manages what: each setting and the one process that reads it (README stack)

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


# ---------------------------------------------------------------- shared helpers for the scripted (timed) animations
TIMED_CSS = """
  .t20 { font-size: 20px; } .t18 { font-size: 18px; } .t16 { font-size: 16px; } .t15 { font-size: 15px; }
  .t14 { font-size: 14px; } .t13 { font-size: 13.5px; }
  .c-white { fill: #ffffff; } .halo { fill: none; stroke: #cf222e; stroke-width: 4px; }
  .ring { fill: none; stroke: #cf222e; stroke-width: 2.5px; transform-box: fill-box; transform-origin: center; }
  .s-orange { stroke: #e07b00; } .s-green { stroke: #4c8c00; } .s-blue { stroke: #1f6feb; }
  .still { display: none; }
  @media (prefers-reduced-motion: reduce) { .still { display: inline; } }"""


class Timeline:
    """Times in seconds on one looping clock of `cyc` seconds; every element keeps its own @keyframes."""

    def __init__(self, cyc):
        self.cyc, self.css = cyc, TIMED_CSS

    def pct(self, s):
        return 100 * s / self.cyc

    def attr(self, cls, name, extra=""):
        return f'class="{cls} motion" style="animation: {name} {self.cyc:g}s linear infinite; opacity: 0;{extra}"'

    def track(self, name, pts):
        """Move a mark through (seconds, x, y, opacity) points; hidden before the first and after the last."""
        hidden = f"opacity: 0; transform: translate({pts[0][1]}px, {pts[0][2]}px);"
        stops = [(0, hidden), (max(self.pct(pts[0][0]) - 0.01, 0), hidden)]
        stops += [(self.pct(s), f"opacity: {o}; transform: translate({x}px, {y}px);") for s, x, y, o in pts]
        stops.append((100, f"opacity: 0; transform: translate({pts[-1][1]}px, {pts[-1][2]}px);"))
        self.css += keyframes(name, stops)

    def shown(self, name, t0, t1, ramp=0.25, peak=1):
        p = self.pct
        self.css += keyframes(name, [(0, "opacity: 0;"), (p(t0), "opacity: 0;"), (p(t0 + ramp), f"opacity: {peak};"),
                                     (p(t1 - ramp), f"opacity: {peak};"), (p(t1), "opacity: 0;"), (100, "opacity: 0;")])

    def ring(self, name, t0, pulses=2):
        stops = [(0, "opacity: 0; transform: scale(1);")]
        for k in range(pulses):
            a = t0 + 0.15 + k * 1.1
            stops += [(self.pct(a), "opacity: 0; transform: scale(1);"),
                      (self.pct(a + 0.02), "opacity: 0.9; transform: scale(1);"),
                      (self.pct(a + 0.9), "opacity: 0; transform: scale(2.1);")]
        stops.append((100, "opacity: 0; transform: scale(1);"))
        self.css += keyframes(name, stops)

    def fill_bar(self, name, t0, t1, hold):
        p = self.pct
        self.css += keyframes(name, [(0, "transform: scaleX(0);"), (p(t0), "transform: scaleX(0);"),
                                     (p(t1), "transform: scaleX(1);"), (p(hold), "transform: scaleX(1);"),
                                     (p(hold) + 0.1, "transform: scaleX(0);"), (100, "transform: scaleX(0);")])


def badge(n, cx, cy, r=13, size="t14"):
    return (f'<circle class="c-red" cx="{cx}" cy="{cy}" r="{r}"/>'
            f'<text class="c-white {size} bold" x="{cx}" y="{cy + 5}" text-anchor="middle">{n}</text>')


def callout_panel(tl, py, items, spans, x0=24, w=912, h=146):
    """One callout per stop in a shared panel: (n, head, lines, chapter) shown during the matching span."""
    out = [f'<rect class="client" x="{x0}" y="{py}" width="{w}" height="{h}" rx="10" stroke-width="1.2"/>']
    for (n, head, lines, chapter), (t0, t1) in zip(items, spans):
        g = [f'<g {tl.attr("", f"co{n}")}>', badge(n, x0 + 38, py + 38, r=19, size="t18"),
             f'<text class="boxink t18 bold" x="{x0 + 72}" y="{py + 44}">{head}</text>']
        g += [f'<text class="boxink t15" x="{x0 + 72}" y="{py + 72 + k * 23}">{line}</text>' for k, line in enumerate(lines)]
        g.append(f'<text class="c-red t14 bold" x="{x0 + w - 16}" y="{py + 44}" text-anchor="end">{chapter}</text></g>')
        out.append("".join(g))
        tl.shown(f"co{n}", t0, t1)
    return out


# ---------------------------------------------------------------- 4. request journey (README hero)
def request_journey():
    W, H = 960, 548
    tl = Timeline(22.0)
    YT, YB = 128, 208           # worker row
    PY = 360                    # callout panel top

    def label(x, title, sub):
        return (f'<text class="boxink t16 bold" x="{x}" y="{YT + 36}" text-anchor="middle">{title}</text>'
                f'<text class="boxink t13" x="{x}" y="{YT + 56}" text-anchor="middle">{sub}</text>')

    body = [MARKER,
            '<text class="ink t20 bold" x="24" y="34">One request: where TTFT and TPOT are spent, and which knob moves each</text>',
            '<text class="muted t15" x="24" y="56">Each stop: the time we measured there and a knob we turned. Stage times '
            'overlap, so they do not add up.</text>']
    # where each user-facing metric accrues
    body += ['<path class="s-orange" d="M160,92 V84 H762 V92" fill="none" stroke-width="2"/>',
             '<text class="c-orange t14 bold" x="461" y="78" text-anchor="middle">TTFT: until the first token '
             '(never under ~43 ms here, even for a 256-token prompt)</text>',
             '<path class="s-green" d="M770,92 V84 H936 V92" fill="none" stroke-width="2"/>',
             '<text class="c-green t13 bold" x="853" y="78" text-anchor="middle">TPOT: per later token</text>']
    glow = {"gttft": ("s-orange", "M160,92 V84 H762 V92", 1.0, 11.6), "gtpot": ("s-green", "M770,92 V84 H936 V92", 11.6, 15.0)}
    for name, (cls, d, t0, t1) in glow.items():
        body.append(f'<path {tl.attr(cls, name)} d="{d}" fill="none" stroke-width="5"/>')
        tl.shown(name, t0, t1, peak=0.8)
    # boxes
    body += [f'<rect class="client" x="24" y="{YT}" width="104" height="80" rx="8" stroke-width="1.6"/>',
             label(76, "Client", "AIPerf"),
             f'<rect class="dynamo" x="160" y="{YT}" width="270" height="80" rx="8" stroke-width="1.6"/>',
             f'<line class="line" x1="295" y1="{YT + 12}" x2="295" y2="{YB - 12}" stroke-width="1" stroke-dasharray="3 3"/>',
             label(227, "Frontend", "HTTP, tokenize"), label(362, "KV router", "picks a worker"),
             f'<rect class="gpu" x="470" y="{YT}" width="160" height="80" rx="8" stroke-width="1.6"/>',
             label(550, "Prefill worker", "GPU 0 · vLLM"),
             f'<rect class="gpu" x="770" y="{YT}" width="166" height="80" rx="8" stroke-width="1.6"/>',
             label(853, "Decode worker", "GPU 1 · vLLM")]
    # paths: prefill leg, KV pull, decode leg, token stream back through the frontend
    body += [arrow(128, YT + 20, 158, YT + 20), arrow(430, YT + 20, 468, YT + 20),
             arrow(468, YT + 60, 432, YT + 60, width=1.3),
             arrow(632, YT + 40, 766, YT + 40, cls="s-purple", width=2.6, dash="7 4"),
             f'<text class="c-purple t13 bold" x="700" y="{YT + 28}" text-anchor="middle">KV cache via NIXL</text>',
             f'<polyline class="line" points="410,{YB} 410,{YB + 15} 800,{YB + 15} 800,{YB + 2}" fill="none" '
             'stroke-width="1.3" marker-end="url(#head)"/>',
             f'<text class="muted t13" x="700" y="{YB + 11}" text-anchor="middle">decode leg</text>',
             f'<polyline class="line" points="853,{YB} 853,{YB + 30} 227,{YB + 30} 227,{YB + 2}" fill="none" '
             'stroke-width="1.6" marker-end="url(#head)"/>',
             arrow(160, YT + 68, 130, YT + 68)]
    # planner
    YP = 280
    body += [f'<rect class="dynamo" x="470" y="{YP}" width="466" height="58" rx="8" stroke-width="1.6"/>',
             badge(5, 494, YP + 29),
             f'<text class="boxink t16 bold" x="518" y="{YP + 24}">Dynamo planner</text>',
             f'<text class="boxink t13" x="518" y="{YP + 44}">resizes the pools · 2 to 3 min per move</text>',
             f'<text class="boxink t13" x="770" y="{YP + 24}">new pod: cold start</text>',
             f'<rect class="client" x="770" y="{YP + 33}" width="150" height="10" rx="3" stroke-width="1"/>',
             arrow(560, YP, 560, YB + 2, dash="4 3", width=1.3), arrow(900, YP, 900, YB + 2, dash="4 3", width=1.3)]
    # badges with the measured time at each stage (always visible, so the still image is complete)
    centers = {1: (172, 112), 2: (482, 112), 3: (646, YT + 66), 4: (800, 112), 5: (494, YP + 29)}
    tags = {1: "~0.1 ms to route", 2: "~70 ms per 3k-token prompt", 3: "5.4 to ~93 ms", 4: "~7 to 10 ms/step"}
    for n, tag in tags.items():
        cx, cy = centers[n]
        body += [badge(n, cx, cy), f'<text class="ink t14 bold" x="{cx + 19}" y="{cy + 5}">{tag}</text>']
    # legend
    for i, (shape, text) in enumerate((('<circle class="c-blue" cx="32" cy="{y}" r="6"/>', "the prompt"),
                                       ('<rect class="c-purple" x="27" y="{y0}" width="10" height="10" rx="2"/>', "KV-cache blocks"),
                                       ('<circle class="c-green" cx="32" cy="{y}" r="6"/>', "output tokens"),
                                       ('<rect class="halo" x="25" y="{y0}" width="14" height="10" rx="3" style="stroke-width: 2.5px;"/>', "stage in focus"))):
        y = 286 + i * 18
        body += [shape.format(y=y, y0=y - 5), f'<text class="muted t14" x="48" y="{y + 5}">{text}</text>']

    # timeline (s): 1 frontend+router, 2 prefill, 3 KV transfer, 4 decode, 5 planner, then the end-to-end scoreboard
    stops = [(1, 1.0, 4.6), (2, 4.8, 8.2), (3, 8.2, 11.6), (4, 11.6, 15.0), (5, 15.0, 18.4)]
    halos = {1: (160, YT, 270, 80), 2: (470, YT, 160, 80), 3: (634, YT + 10, 132, 36), 4: (770, YT, 166, 80),
             5: (470, YP, 466, 58)}
    for n, t0, t1 in stops:
        x, y, w, h = halos[n]
        body.append(f'<rect {tl.attr("halo", f"hl{n}")} x="{x - 3}" y="{y - 3}" width="{w + 6}" height="{h + 6}" rx="10"/>')
        tl.shown(f"hl{n}", t0, t1, peak=0.9)
        cx, cy = centers[n]
        body.append(f'<circle {tl.attr("ring", f"rg{n}")} cx="{cx}" cy="{cy}" r="13"/>')
        tl.ring(f"rg{n}", t0)
    # the prompt: client -> frontend -> router -> prefill
    body.append(f'<circle {tl.attr("c-blue", "prompt")} r="7" cx="0" cy="0"/>')
    tl.track("prompt", [(0.1, 128, YT + 20, 1), (0.9, 154, YT + 20, 1), (4.6, 154, YT + 20, 1), (4.65, 160, YT, 1),
                        (4.9, 430, YT, 1), (4.95, 436, YT + 20, 1), (5.0, 464, YT + 20, 1), (8.2, 464, YT + 20, 1), (8.21, 464, YT + 20, 0)])
    # the frontend sends the decode leg; the decode worker then pulls the KV cache as many small copies
    body.append(f'<circle {tl.attr("c-blue", "leg2")} r="6" cx="0" cy="0"/>')
    tl.track("leg2", [(8.25, 468, YT + 60, 1), (8.45, 432, YT + 60, 1), (8.5, 410, YB - 2, 1), (8.6, 410, YB + 15, 1),
                      (8.9, 800, YB + 15, 1), (9.0, 800, YB + 2, 1), (9.01, 800, YB + 2, 0)])
    for i in range(10):
        a = 9.05 + i * 0.21
        body.append(f'<rect {tl.attr("c-purple", f"kv{i}")} x="-4" y="-4" width="8" height="8" rx="1.5"/>')
        tl.track(f"kv{i}", [(a, 634, YT + 40, 1), (a + 0.5, 764, YT + 40, 1), (a + 0.51, 764, YT + 40, 0)])
    # output tokens stream back through the frontend
    for i in range(5):
        a = 11.8 + i * 0.5
        body.append(f'<circle {tl.attr("c-green", f"tok{i}")} r="6" cx="0" cy="0"/>')
        tl.track(f"tok{i}", [(a, 853, YB + 2, 1), (a + 0.2, 853, YB + 30, 1), (a + 1.0, 227, YB + 30, 1),
                             (a + 1.15, 227, YB + 6, 1), (a + 1.3, 158, YT + 68, 1), (a + 1.4, 130, YT + 68, 1),
                             (a + 1.41, 130, YT + 68, 0)])
    # planner: a new pod's cold start fills the bar
    body.append(f'<rect class="c-blue motion" x="770" y="{YP + 33}" width="150" height="10" rx="3" '
                f'style="transform-origin: 770px {YP + 33}px; animation: cold {tl.cyc:g}s linear infinite; transform: scaleX(0);"/>')
    tl.fill_bar("cold", 15.2, 18.0, 18.3)

    # callouts: what the stage sets, the time we measured there, the knob we turned and what it did
    items = [
        (1, "Frontend and router: decide where the prompt runs (TTFT)",
         ["Time: the routing decision takes ~0.1 ms when nothing queues.",
          "Knob: --router-prefill-load-scale 10: prefix hit rate 0.64 → 0.89, median TTFT unchanged at 44.6 ms.",
          "Knob: --router-queue-threshold 0.5 under overload: TTFT p90 352 → 144 ms, ~34 ms router wait, TPOT +8%."],
         "Part 3"),
        (2, "Prefill: sets TTFT; on a shared GPU it also sets TPOT",
         ["Time: a 3,000-token prompt keeps an H100 busy ~70 ms (Nsight, upper bound); one user's TTFT is 79 ms.",
          "On one shared GPU at 64 users: one token gap in ten is ~200 ms, the median ~17 ms.",
          "Knob (shared GPU): --max-num-batched-tokens 512 at 64 users: TPOT 43.7 → 17.6 ms, TTFT p50 0.6 → 4.8 s."],
         "Parts 2 and 6"),
        (3, "KV transfer: adds to TTFT, and posting it stalls running decodes",
         ["Time, 6,000-token KV cache (846 MiB): 5.4 ms idle NVLink, ~93 ms loaded (~51 ms posting), ~2 s TCP.",
          "Knob: NVLink, not TCP (both GPUs visible, a diagnostic setup): TTFT p50 16.1 s → 205 ms at 16 users.",
          "Knob: --block-size 64: loaded transfer ~93 → ~26 ms, but TTFT p50 at 64 users 2.9 → 3.3 s (a partial fix)."],
         "Part 4"),
        (4, "Decode: sets TPOT, one engine step per token",
         ["Time: a step reads the weights and KV cache: 6.85 ms at 1 user, 9.94 ms at 64 (short prompts).",
          "Knob: FP8 weights: TPOT 7.59 → 5.41 ms (-29%) at 16 short-prompt users; accuracy not evaluated.",
          "Knob: prefill/decode split: TPOT 23.8 → 12.7 ms at 64 prompt-heavy users (2 GPUs)."],
         "Parts 2, 4 and 6"),
        (5, "Planner: sets capacity, in minutes rather than seconds",
         ["Time: each move deletes a pod and cold-starts another: 1 min 51 s to 2 min 49 s.",
          "Setting (not varied): load thresholds checked every 10 s. A pod was cold-starting ~75% of the 5-minute",
          "prefill phase (1,500-token prompts, 48 users, MIG slices on 2x H100)."],
         "Part 5"),
    ]
    body += callout_panel(tl, PY, items, [(t0, t1) for _, t0, t1 in stops])

    # end of the loop: the measured end-to-end trade, the only numbers here that belong to one run each
    sb = [f'<text class="boxink t18 bold" x="44" y="{PY + 34}">End to end: measured TTFT p50 and TPOT, prompt-heavy '
          'traffic on 2 H100s</text>',
          f'<text class="c-red t14 bold" x="920" y="{PY + 34}" text-anchor="end">Part 4</text>',
          f'<text class="muted t14 bold" x="350" y="{PY + 62}">16 users: TTFT p50 · TPOT</text>',
          f'<text class="muted t14 bold" x="620" y="{PY + 62}">64 users: TTFT p50 · TPOT</text>']
    rows = [("Two aggregated GPUs + KV router", "449 ms · 9.6 ms", "603 ms · 23.8 ms"),
            ("Prefill/decode split", "205 ms · 8.8 ms", "2,878 ms · 12.7 ms")]
    for k, (name, a, b) in enumerate(rows):
        y = PY + 86 + k * 24
        sb += [f'<text class="boxink t15 bold" x="44" y="{y}">{name}</text>',
               f'<text class="boxink t15" x="350" y="{y}">{a}</text>', f'<text class="boxink t15" x="620" y="{y}">{b}</text>']
    sb.append(f'<text class="boxink t15" x="44" y="{PY + 136}">16 users: the split wins both at p50 (p99 TTFT 1,234 vs '
              '594 ms). 64 users: TPOT halves, TTFT grows 4.8x, req/s -24%.</text>')
    body.append(f'<g {tl.attr("", "score")}>' + "".join(sb) + "</g>")
    tl.shown("score", 18.4, tl.cyc - 0.05)
    body.append(f'<g {tl.attr("", "rgall")}>' + "".join(
        f'<circle class="ring" cx="{cx}" cy="{cy}" r="15.5" style="stroke-width: 2px;"/>' for cx, cy in centers.values()) + "</g>")
    tl.shown("rgall", 18.4, tl.cyc - 0.05, peak=0.9)

    # still image (reduced motion): the knob at each stop
    still = [(1, "Router: --router-prefill-load-scale 10, hit rate 0.64 → 0.89 (Part 3)"),
             (2, "Prefill: chunk budget 512, one GPU, 64 users: TPOT 43.7 → 17.6 ms, TTFT p50 0.6 → 4.8 s (Part 6)"),
             (3, "KV transfer: NVLink instead of TCP; --block-size 64, a partial fix (Part 4)"),
             (4, "Decode: FP8 weights, TPOT -29% at 16 users; prefill/decode split, TPOT 23.8 → 12.7 ms (Parts 4, 6)"),
             (5, "Planner: 1 min 51 s to 2 min 49 s per move (Part 5)")]
    g = ['<g class="still">']
    for i, (n, text) in enumerate(still):
        y = PY + 30 + i * 25
        g += [badge(n, 54, y - 5, r=11, size="t13"), f'<text class="boxink t15" x="74" y="{y}">{text}</text>']
    body.append("".join(g) + "</g>")

    body.append(f'<text class="muted t13" x="24" y="{H - 14}">Qwen3-8B on H100 SXM (routing: Qwen3-4B, same KV size per '
                'token). Dots show order, not duration.</text>')
    return svg(W, H, "\n".join(body), tl.css)


# ---------------------------------------------------------------- 5. who manages what (README stack)
def stack_ownership():
    W, H = 960, 700
    tl = Timeline(36.0)
    tl.css += "\n  .card { fill: #ffffff; stroke: #d0d7de; } .platform { fill: #e6edf3; stroke: #6e7781; }"
    tl.css += "\n  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }"
    Y0, RH, GAP = 88, 44, 6
    CX, CW, LX, LW = 24, 368, 428, 344        # config card column, process column
    PY = 552                                  # callout panel top
    # (setting you write, where you write it, process that reads it, what it decides, box class)
    rows = [
        ("optimization_target: load", "DGD, planner component", "Dynamo planner", "how pool sizes change over time", "dynamo"),
        ("replicas: 2", "DGD, worker component", "Dynamo operator", "which pods exist, and how many", "dynamo"),
        ("--router-prefill-load-scale 10", "DGD, frontend args", "Dynamo frontend (router)", "which worker gets each request", "dynamo"),
        ("--disaggregation-mode prefill", "DGD, worker args", "dynamo.vllm wrapper", "prefill, decode, or both", "gpu"),
        ("--max-num-batched-tokens 512", "DGD, worker args (passed to vLLM)", "vLLM scheduler", "how each engine step is formed", "gpu"),
        ("(not set)", "nowhere: picked per GPU", "vLLM source code", "every default you did not set", "gpu"),
        ("--kv-transfer-config, UCX_* env", "DGD, worker args and env", "NIXL connector, NIXL, UCX", "how KV moves, and over which wire", "kvbox"),
        ("nvidia.com/gpu: 1", "pod spec, resources.limits", "KAI scheduler, kubelet, device plugin", "which node and GPU a pod lands on", "platform"),
        ("nvidia.com/mig.config=all-3g.40gb", "node label, Helm values", "GPU Operator", "which GPUs exist, and how they are split", "platform"),
    ]
    top = lambda i: Y0 + i * (RH + GAP)
    mid = lambda i: top(i) + RH / 2
    body = [MARKER,
            '<text class="ink t20 bold" x="24" y="34">Who manages what: every setting is read by exactly one process</text>',
            '<text class="muted t15" x="24" y="56">That process decides what the setting means; the YAML file you wrote it '
            'in does not.</text>',
            f'<text class="muted t13 bold" x="{CX}" y="{Y0 - 10}">WHAT YOU WRITE, AND WHERE</text>',
            f'<text class="muted t13 bold" x="{LX}" y="{Y0 - 10}">THE PROCESS THAT READS IT</text>',
            f'<text class="muted t13 bold" x="786" y="{Y0 - 10}">WHERE IT RUNS</text>']
    for i, (cfg, where, who, decides, cls) in enumerate(rows):
        y = top(i)
        body += [f'<rect class="card" x="{CX}" y="{y}" width="{CW}" height="{RH}" rx="6" stroke-width="1"/>',
                 badge(i + 1, CX + 20, y + RH / 2, r=11, size="t13"),
                 f'<text class="boxink t13 mono bold" x="{CX + 40}" y="{y + 19}">{cfg}</text>',
                 f'<text class="muted small" x="{CX + 40}" y="{y + 36}">{where}</text>',
                 arrow(CX + CW + 2, mid(i), LX - 2, mid(i)),
                 f'<rect class="{cls}" x="{LX}" y="{y}" width="{LW}" height="{RH}" rx="6" stroke-width="1.4"/>',
                 f'<text class="boxink t14 bold" x="{LX + 14}" y="{y + 19}">{who}</text>',
                 f'<text class="boxink t13" x="{LX + 14}" y="{y + 36}">decides {decides}</text>']
    # where each process runs
    groups = [(0, 1, "Dynamo control pods", "operator; planner per graph"), (2, 2, "Frontend pod", "one per graph"),
              (3, 6, "vLLM worker pod", "one per GPU or MIG slice"), (7, 8, "Kubernetes cluster", "KAI, kubelet, GPU Operator")]
    for a, b, name, sub in groups:
        y1, y2 = top(a) + 4, top(b) + RH - 4
        ym = (y1 + y2) / 2
        body += [f'<path class="line" d="M780,{y1} H786 V{y2} H780" fill="none" stroke-width="1.6"/>',
                 f'<text class="ink t14 bold" x="798" y="{ym - 2}">{name}</text>',
                 f'<text class="muted small" x="798" y="{ym + 15}">{sub}</text>']

    # one step per row: the setting and its reader light up, a dot runs the arrow, the callout explains
    T0, STEP = 0.6, 3.4
    items = [
        (1, "Dynamo planner: how pool sizes change over time",
         ["Set in the DGD's planner component. Each decision becomes a pod deleted and another cold-started:",
          "1 min 51 s to 2 min 49 s per move."], "Part 5"),
        (2, "Dynamo operator: which pods exist, and how many",
         ["Set in the DGD's components, replicas and podTemplate.",
          "It watches the DynamoGraphDeployment and creates or deletes pods to match it."], "Part 1"),
        (3, "Dynamo frontend: which worker gets each request",
         ["Set in the frontend container's args, alongside --router-mode kv.",
          "--router-prefill-load-scale 10 raised the H100 prefix hit rate 0.64 → 0.89 (Qwen3-4B)."], "Part 3"),
        (4, "dynamo.vllm wrapper: prefill, decode, or both",
         ["Set in the worker's args. Dynamo and vLLM exchange one small dictionary, kv_transfer_params;",
          "Dynamo never touches the KV bytes."], "Part 4"),
        (5, "vLLM scheduler: how each engine step is formed",
         ["Set in the worker's args, passed through to vLLM. At 64 users, --max-num-batched-tokens 512 took",
          "TPOT 43.7 → 17.6 ms and TTFT p50 0.6 → 4.8 s."], "Part 6"),
        (6, "vLLM source code: every default you did not set",
         ["Nothing in any YAML. The default depends on the GPU: the chunk budget is 8,192 tokens",
          "on an H100 80GB and 2,048 on an A100 40GB."], "Parts 1 and 2"),
        (7, "NIXL connector, NIXL, UCX: how the KV cache moves",
         ["Set in the worker's args and env; UCX reads UCX_* variables and picks the wire itself, silently.",
          "NVLink moved 846 MiB in 5.4 ms; TCP took ~2 s."], "Part 4"),
        (8, "Scheduler, kubelet, device plugin: which node and GPU a pod lands on",
         ["Set in the pod's resources.limits. With nvidia.com/gpu: 1 the device plugin gives each pod exactly one GPU,",
          "so the decode pod cannot open the prefill GPU's memory and UCX falls back to TCP."], "Part 4"),
        (9, "GPU Operator: which GPUs exist, and how they are split",
         ["Set in Helm values and node labels. nvidia.com/mig.config=all-3g.40gb turns",
          "2 H100s into 4 schedulable slices."], "Part 5"),
    ]
    spans = []
    for i in range(len(rows)):
        t0, t1 = T0 + i * STEP, T0 + (i + 1) * STEP
        spans.append((t0, t1))
        y = top(i)
        for name, x, w in ((f"hc{i}", CX, CW), (f"hp{i}", LX, LW)):
            body.append(f'<rect {tl.attr("halo", name)} x="{x - 3}" y="{y - 3}" width="{w + 6}" height="{RH + 6}" rx="8"/>')
            tl.shown(name, t0, t1, peak=0.9)
        body.append(f'<circle {tl.attr("c-red", f"dot{i}")} r="5" cx="0" cy="0"/>')
        tl.track(f"dot{i}", [(t0 + 0.1, CX + CW, mid(i), 1), (t0 + 0.6, LX, mid(i), 1), (t0 + 0.61, LX, mid(i), 0)])
        body.append(f'<circle {tl.attr("ring", f"rg{i}")} cx="{CX + 20}" cy="{mid(i)}" r="11"/>')
        tl.ring(f"rg{i}", t0)
    body += callout_panel(tl, PY, items, spans, h=118)
    end_text = [f'<text class="boxink t18 bold" x="44" y="{PY + 40}">When a number moves unexpectedly, walk this list</text>',
                f'<text class="boxink t15" x="44" y="{PY + 68}">Which process read the setting, and what did it log at '
                'startup?',
                f'<text class="boxink t15" x="44" y="{PY + 91}">Part 1 has the full table, with the reasoning behind '
                'each row.</text>',
                f'<text class="c-red t14 bold" x="920" y="{PY + 40}" text-anchor="end">Part 1</text>']
    end_text[1] += "</text>"
    t_end = T0 + len(rows) * STEP
    body.append(f'<g {tl.attr("", "end")}>' + "".join(end_text) + "</g>")
    tl.shown("end", t_end, tl.cyc - 0.05)
    body.append('<g class="still"><rect class="client" x="24" y="%d" width="912" height="118" rx="10" '
                'stroke-width="1.2"/>%s</g>' % (PY, "".join(end_text)))
    body.append(f'<text class="muted t13" x="24" y="{H - 14}">DGD: DynamoGraphDeployment, the one Kubernetes resource '
                'that describes a Dynamo serving graph.</text>')
    return svg(W, H, "\n".join(body), tl.css)


if __name__ == "__main__":
    for name, fn in (("anim-request-path", request_path), ("anim-interference", interference),
                     ("anim-kv-transfer", kv_transfer), ("anim-request-journey", request_journey),
                     ("anim-stack-ownership", stack_ownership)):
        p = OUT / f"{name}.svg"
        p.write_text(fn(), encoding='utf-8')
        print("wrote", p.relative_to(OUT.parent), f"{p.stat().st_size / 1024:.1f} KB")
