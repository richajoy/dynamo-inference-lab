#!/usr/bin/env python3
"""Generate the animated explanatory diagrams in figures/ (self-contained SVG, no dependencies).

  figures/anim-agg-vs-disagg.svg  same traffic on two H100s: aggregated workers vs a prefill/decode split
  figures/anim-interference.svg   why a prompt chunk stalls every decoding user, and what disaggregation changes
  figures/anim-kv-transfer.svg    why a KV transfer over NVLink slows under load: contiguous vs fragmented KV blocks
  figures/anim-request-journey.svg  one request: where TTFT and TPOT are spent, and the knob that moves each (README hero)
  figures/anim-stack-overview.svg   the stack and who manages it (README): stack-overview.mmd plus the response path

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


# ---------------------------------------------------------------- 1. interference timeline
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


# ---------------------------------------------------------------- 2. KV transfer
def kv_transfer():
    W, H, CYC = 960, 300, 6
    css = ""
    body = [MARKER, '<text class="ink title" x="24" y="34">Why a KV transfer over NVLink slows down under load</text>',
            '<text class="muted small" x="24" y="54">Measured on 2x H100: an idle 846 MiB transfer took 5.4 ms; under load a '
            '6,000-token (843.75 MiB) transfer took ~93 ms.</text>']
    # descriptor panel
    yb = 100
    body.append(f'<text class="ink label bold" x="24" y="{yb - 8}">How the KV blocks are laid out decides the copy list</text>')
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
    body.append('<text class="muted small" x="24" y="270">Each descriptor costs ~3.8 μs of CPU work to post. 13,500 of them '
                'took more than half of the ~93 ms.</text>')
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
    """Times in seconds on one looping clock of `cyc` seconds; every element keeps its own @keyframes.

    `pace` optionally slows parts of the loop: [(until, factor), ...] in script seconds, so motion can stay brisk while
    the time spent on a callout is stretched for reading. Script times are mapped through it; `dur` is the real loop."""

    def __init__(self, cyc, pace=None):
        self.cyc, self.css, self.pace = cyc, TIMED_CSS, pace or [(cyc, 1)]
        self.dur = self.real(cyc)

    def real(self, s):
        out, prev = 0.0, 0.0
        for until, k in self.pace:
            out += (min(s, until) - prev) * k
            if s <= until:
                return out
            prev = until
        return out + (s - prev) * self.pace[-1][1]

    def pct(self, s):
        return 100 * self.real(s) / self.dur

    def attr(self, cls, name, extra=""):
        return f'class="{cls} motion" style="animation: {name} {self.dur:.2f}s linear infinite; opacity: 0;{extra}"'

    def track(self, name, pts):
        """Move a mark through (seconds, x, y, opacity) points; hidden before the first and after the last."""
        hidden = f"opacity: 0; transform: translate({pts[0][1]}px, {pts[0][2]}px);"
        stops = [(0, hidden), (max(self.pct(pts[0][0]) - 0.01, 0), hidden)]
        stops += [(self.pct(s), f"opacity: {o}; transform: translate({x}px, {y}px);") for s, x, y, o in pts]
        stops.append((100, f"opacity: 0; transform: translate({pts[-1][1]}px, {pts[-1][2]}px);"))
        self.css += keyframes(name, stops)

    def shown(self, name, t0, t1, ramp=0.25, peak=1):
        """Fade in at t0 and out by t1 (script seconds); the fades last `ramp` real seconds however slow the pace."""
        a, b, r = self.pct(t0), self.pct(t1), 100 * ramp / self.dur
        self.css += keyframes(name, [(0, "opacity: 0;"), (a, "opacity: 0;"), (a + r, f"opacity: {peak};"),
                                     (b - r, f"opacity: {peak};"), (b, "opacity: 0;"), (100, "opacity: 0;")])

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


# ---------------------------------------------------------------- 3. request journey (README hero)
def request_journey():
    W, H = 960, 548
    # Motion runs at 1.2x script time; the quiet end of each stop stretches so its callout holds ~11 s to be read,
    # and the end-to-end scoreboard ~13 s. No two moving marks fall inside a stretched span.
    tl = Timeline(22.0, pace=[(3.2, 1.2), (4.6, 5.97), (7.0, 1.2), (8.2, 6.97), (11.1, 1.2), (11.6, 15.04),
                              (14.5, 1.2), (15.0, 15.04), (18.0, 2.2), (18.4, 11.0), (22.0, 3.61)])
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
             f'<text class="boxink t13" x="518" y="{YP + 44}">resizes the pools · about 2 to 3 min per move</text>',
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
        a = 11.8 + i * 0.3
        body.append(f'<circle {tl.attr("c-green", f"tok{i}")} r="6" cx="0" cy="0"/>')
        tl.track(f"tok{i}", [(a, 853, YB + 2, 1), (a + 0.2, 853, YB + 30, 1), (a + 1.0, 227, YB + 30, 1),
                             (a + 1.15, 227, YB + 6, 1), (a + 1.3, 158, YT + 68, 1), (a + 1.4, 130, YT + 68, 1),
                             (a + 1.41, 130, YT + 68, 0)])
    # planner: a new pod's cold start fills the bar
    body.append(f'<rect class="c-blue motion" x="770" y="{YP + 33}" width="150" height="10" rx="3" '
                f'style="transform-origin: 770px {YP + 33}px; animation: cold {tl.dur:.2f}s linear infinite; transform: scaleX(0);"/>')
    tl.fill_bar("cold", 15.2, 18.0, 18.38)

    # callouts: what the stage sets, the time we measured there, the knob we turned and what it did
    items = [
        (1, "Frontend and router: decide where the prompt runs (TTFT)",
         ["Time: the routing decision takes ~0.1 ms when nothing queues.",
          "Knob: --router-prefill-load-scale 10: prefix hit rate 0.64 → 0.89, median TTFT unchanged at 44.6 ms.",
          "Knob: --router-queue-threshold 0.5 under overload: TTFT p90 352 → 144 ms, ~34 ms router wait, TPOT +8%."],
         "Part 3"),
        (2, "Prefill: sets TTFT; on a shared GPU it also sets TPOT",
         ["Time: a 3,000-token prompt keeps an H100 busy ~70 ms (Nsight, upper bound); one user's TTFT is 79 ms.",
          "On one shared GPU at 64 prompt-heavy users: one token gap in ten is ~200 ms, the median ~17 ms.",
          "Knob: --max-num-batched-tokens 512, same GPU and users: TPOT 43.7 → 17.6 ms, TTFT p50 0.6 → 4.8 s."],
         "Parts 2 and 6"),
        (3, "KV transfer: adds to TTFT, and posting it stalls running decodes",
         ["Time, one ~6,000-token KV cache over NVLink: 5.4 ms idle, ~93 ms under load.",
          "Why: under load it is ~13,500 scattered spans, each posted by the CPU in ~3.8 μs: ~51 ms of the ~93.",
          "Knob: --block-size 64: transfer ~93 → ~26 ms; 64 prompt-heavy users (57 in flight): TPOT -20%, TTFT p50 +15%."],
         "Part 4"),
        (4, "Decode: sets TPOT, one engine step per token",
         ["Time: a step reads the weights and KV cache: 6.85 ms at 1 user, 9.94 ms at 64 (short prompts, one GPU).",
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
             (2, "Prefill: chunk budget 512, one GPU, 64 prompt-heavy users: TPOT 43.7 → 17.6 ms, TTFT p50 0.6 → 4.8 s (Part 6)"),
             (3, "KV transfer: --block-size 64, ~93 → ~26 ms per loaded transfer, a partial fix (Part 4)"),
             (4, "Decode: FP8 weights, TPOT -29% at 16 short-prompt users; prefill/decode split, TPOT 23.8 → 12.7 ms (Parts 4, 6)"),
             (5, "Planner: 1 min 51 s to 2 min 49 s per move (Part 5)")]
    g = ['<g class="still">']
    for i, (n, text) in enumerate(still):
        y = PY + 30 + i * 25
        g += [badge(n, 54, y - 5, r=11, size="t13"), f'<text class="boxink t15" x="74" y="{y}">{text}</text>']
    body.append("".join(g) + "</g>")

    body.append(f'<text class="muted t13" x="24" y="{H - 14}">Qwen3-8B on H100 SXM (routing: Qwen3-4B, same KV size per '
                'token). Dots show order, not duration.</text>')
    return svg(W, H, "\n".join(body), tl.css)


# ---------------------------------------------------------------- 4. aggregated vs disaggregated (README, Part 4)
def agg_vs_disagg():
    """Same traffic on the same two H100s: two aggregated workers against 1 prefill + 1 decode. Steps are schematic;
    every number is a measured result from docs/04 (prompt-heavy, 64 concurrent unless it says 16 users)."""
    W, H = 960, 560
    tl = Timeline(40.0)
    X0, XE = 200, 724                       # step tracks
    DW, MW, PW = 22, 88, 62                 # decode step, step with a prompt chunk, one prompt's prefill (schematic)
    A = (0.5, 9.5)                          # playhead sweep, aggregated lane
    B = (13.5, 22.5)                        # playhead sweep, split lane
    END = tl.cyc - 0.05

    def at(x, sweep):
        return sweep[0] + (sweep[1] - sweep[0]) * (x - X0) / (XE - X0)

    def lights(name, t):                    # dim until the playhead reaches it, then lit for the rest of the loop
        p = tl.pct(t)
        tl.css += keyframes(name, [(0, "opacity: 0.18;"), (max(p - 0.01, 0), "opacity: 0.18;"), (p, "opacity: 1;"),
                                   (99, "opacity: 1;"), (100, "opacity: 0.18;")])

    def appears(name, t):                   # hidden until t, then shown; fully visible in the reduced-motion still
        p = tl.pct(t)
        tl.css += keyframes(name, [(0, "opacity: 0;"), (max(p - 0.01, 0), "opacity: 0;"), (p, "opacity: 1;"),
                                   (99, "opacity: 1;"), (100, "opacity: 0;")])

    body = [MARKER,
            '<text class="ink t20 bold" x="24" y="34">Same traffic, same two H100s: aggregated or split?</text>',
            '<text class="muted t15" x="24" y="56">Prompt-heavy traffic (3,000-token prompts, 128 output tokens), 64 '
            'concurrent users, Qwen3-8B. Measured results on the right.</text>']
    legend = [("decode", "decode step"), ("prefill", "prompt work"), ("kvbox", "KV cache"),
              ("wait", "every decoding user on that GPU waits")]
    x = 24
    for cls, text in legend:
        body += [f'<rect class="{cls}" x="{x}" y="68" width="14" height="12" rx="2" stroke-width="1"/>',
                 f'<text class="muted t13" x="{x + 20}" y="79">{text}</text>']
        x += 34 + 7.2 * len(text)

    def steps(y, pattern, name, sweep):
        out, x, k = [], X0, 0
        for kind in pattern:
            w = MW if kind == "M" else DW
            if x + w > XE:
                break
            if kind == "M":
                out += [f'<rect class="prefill" x="{x + 1}" y="{y}" width="{w - 2}" height="20" rx="3" stroke-width="1"/>',
                        f'<rect class="decode" x="{x + 1}" y="{y + 20}" width="{w - 2}" height="8" rx="2" stroke-width="1"/>',
                        f'<rect class="wait" x="{x + 1}" y="{y + 32}" width="{w - 2}" height="12" rx="2" stroke-width="1"/>']
            else:
                out.append(f'<rect class="decode" x="{x + 1}" y="{y}" width="{w - 2}" height="28" rx="3" stroke-width="1"/>')
            x += w
            out.append(f'<circle class="c-green" cx="{x - 4}" cy="{y + 38}" r="4" style="animation: {name}{k} {tl.dur:.2f}s linear infinite;"/>')
            lights(f"{name}{k}", at(x, sweep))
            k += 1
        return out

    # lane A: two aggregated workers
    body.append('<text class="ink t16 bold" x="24" y="106">Aggregated: each GPU prefills and decodes; the KV router spreads '
                'requests</text>')
    for i, (y, pattern) in enumerate(((116, "dddMdddddMdddddddd"), (166, "dddddMddddddMddddd"))):
        body.append(f'<text class="ink t14 bold" x="24" y="{y + 19}">GPU {i}</text>')
        body += steps(y, pattern, f"ta{i}", A)
    body.append('<text class="muted t13" x="24" y="208">tokens: one dot per step</text>')
    # lane B: one prefill GPU, one decode GPU
    body.append('<text class="ink t16 bold" x="24" y="244">Split: GPU 0 only prefills, GPU 1 only decodes; the KV cache '
                'moves over NVLink</text>')
    yp, yd = 254, 310
    body += [f'<text class="ink t14 bold" x="24" y="{yp + 19}">GPU 0 · prefill</text>',
             f'<text class="ink t14 bold" x="24" y="{yd + 19}">GPU 1 · decode</text>',
             f'<text class="muted t13" x="168" y="{yp + 44}" text-anchor="middle">waiting</text>']
    for k in range(int((XE - X0) // PW)):
        x = X0 + k * PW
        body += [f'<rect class="prefill" x="{x + 1}" y="{yp}" width="{PW - 2}" height="28" rx="3" stroke-width="1"/>',
                 f'<text class="boxink t13" x="{x + PW / 2}" y="{yp + 19}" text-anchor="middle">prompt</text>']
        t = at(x + PW, B)
        body.append(f'<rect {tl.attr("c-purple", f"kv{k}")} x="-5" y="-5" width="10" height="10" rx="2"/>')
        tl.track(f"kv{k}", [(t, x + PW - 6, yp + 32, 1), (t + 0.6, x + PW - 6, yd - 6, 1), (t + 0.61, x + PW - 6, yd - 6, 0)])
    for k in range(5):                      # the prefill queue grows: one GPU cannot keep up with 64 users
        body.append(f'<circle class="c-blue" cx="{192 - k * 12}" cy="{yp + 14}" r="4.5" '
                    f'style="animation: q{k} {tl.dur:.2f}s linear infinite;"/>')
        appears(f"q{k}", B[0] + 1.2 + k * 1.6)
    body += steps(yd, "d" * 40, "tb", B)
    # playheads
    for name, (y0, y1), sweep in (("phA", (112, 212), A), ("phB", (250, 352), B)):
        body.append(f'<line {tl.attr("s-red", name)} x1="0" y1="{y0}" x2="0" y2="{y1}" stroke-width="1.5"/>')
        tl.track(name, [(sweep[0], X0, 0, 1), (sweep[1], XE, 0, 1), (sweep[1] + 0.01, XE, 0, 0)])
    # measured results, winner of each metric in green
    for y, rows in ((112, (("TPOT", "23.8 ms", "c-red"), ("TTFT p50", "603 ms", "c-green"), ("throughput", "16.8 req/s", "c-green"))),
                    (250, (("TPOT", "12.7 ms", "c-green"), ("TTFT p50", "2,878 ms", "c-red"), ("throughput", "12.8 req/s", "c-red")))):
        body.append(f'<rect class="client" x="744" y="{y}" width="192" height="100" rx="8" stroke-width="1.2"/>')
        for k, (label, value, cls) in enumerate(rows):
            body += [f'<text class="boxink t14" x="758" y="{y + 28 + k * 28}">{label}</text>',
                     f'<text class="{cls} t15 bold" x="922" y="{y + 28 + k * 28}" text-anchor="end">{value}</text>']
    # stage highlights
    for name, (x, y, w, h), t0, t1 in (("hlA", (16, 90, 928, 128), 0.3, 13.4), ("hlB", (16, 228, 928, 128), 13.5, 27.4),
                                       ("hlM1", (744, 112, 192, 100), 27.5, END), ("hlM2", (744, 250, 192, 100), 27.5, END)):
        body.append(f'<rect {tl.attr("halo", name)} x="{x - 3}" y="{y - 3}" width="{w + 6}" height="{h + 6}" rx="10"/>')
        tl.shown(name, t0, t1, peak=0.9)

    PY = 372
    items = [
        (1, "Aggregated: prompt work rides inside the decode steps",
         ["Each GPU runs prompt chunks in its decode steps, so every user on that GPU waits while a chunk runs.",
          "Both GPUs share the prompts: TTFT p50 603 ms and 16.8 requests/s.",
          "The cost lands on TPOT: 23.8 ms per token, against 12.7 ms with the split."],
         "Parts 2 and 4"),
        (2, "Split: decoding never waits for a prompt, but prompts wait for one GPU",
         ["The decode GPU only decodes: TPOT 12.7 ms, about half.",
          "Every prompt queues for a single prefill GPU and its KV cache then crosses NVLink: TTFT p50 2,878 ms.",
          "The prefill GPU is now the bottleneck: 12.8 requests/s, -24% (58 of 64 requests in flight)."],
         "Part 4"),
        (3, "When the split pays",
         ["At 16 users it wins TTFT p50 (205 vs 449 ms) and TPOT (8.8 vs 9.6 ms) at equal throughput; TTFT p99 doubles.",
          "At 64 users it trades TTFT (4.8x) and throughput (-24%) for half the TPOT.",
          "Use it for long prompts with a TPOT target, KV over NVLink or RDMA, and pools sized to the traffic."],
         "Part 4"),
    ]
    body += callout_panel(tl, PY, items, [(0.3, 13.4), (13.5, 27.4), (27.5, END)])
    still = ["Aggregated: prompt chunks stall decoding. TPOT 23.8 ms, TTFT p50 603 ms, 16.8 requests/s.",
             "Split: decoding never stalls, TPOT 12.7 ms; one prefill GPU queues prompts, TTFT p50 2,878 ms, 12.8 requests/s (58 in flight).",
             "At 16 users the split wins TTFT p50 (205 vs 449 ms) and TPOT (8.8 vs 9.6 ms) at equal throughput.",
             "Use it for long prompts with a TPOT target, KV over NVLink or RDMA, and pools sized to the traffic."]
    body.append('<g class="still">' + "".join(f'<text class="boxink t15" x="44" y="{PY + 32 + i * 28}">{t}</text>'
                                              for i, t in enumerate(still)) + "</g>")
    body.append(f'<text class="muted t13" x="24" y="{H - 14}">Steps drawn schematically, not to scale. Qwen3-8B BF16, '
                '2x H100 SXM, KV block 16; results from Part 4.</text>')
    return svg(W, H, "\n".join(body), tl.css)


# ---------------------------------------------------------------- 5. who manages what (README stack)
def rounded(points, r=8):
    """SVG path through `points` with rounded corners, like Mermaid's flowchart edges."""
    d = [f"M{points[0][0]},{points[0][1]}"]
    for (x0, y0), (x1, y1), (x2, y2) in zip(points, points[1:], points[2:]):
        a = max(abs(x1 - x0), abs(y1 - y0)) or 1
        b = max(abs(x2 - x1), abs(y2 - y1)) or 1
        k1, k2 = min(r, a / 2) / a, min(r, b / 2) / b
        d.append(f"L{x1 - (x1 - x0) * k1:g},{y1 - (y1 - y0) * k1:g} Q{x1},{y1} {x1 + (x2 - x1) * k2:g},{y1 + (y2 - y1) * k2:g}")
    d.append(f"L{points[-1][0]},{points[-1][1]}")
    return " ".join(d)


def stack_overview():
    """The README stack diagram (figures/diagrams/stack-overview.mmd), same boxes, edges and labels, plus the response
    path, animated: the planner and operator create the worker pods, then one request flows through them and its
    output tokens stream back through the frontend to the client."""
    W, H, CYC = 560, 510, 14
    css = ("\n  .m15 { font-size: 15px; } .m14 { font-size: 14px; }"
           "\n  .chip { fill: #e8e8e8; }"
           "\n  .e-blue { stroke: #1f6feb; fill: none; } .e-grey { fill: none; } .e-kv { stroke: #8250df; fill: none; }"
           "\n  .e-green { stroke: #4c8c00; fill: none; }")
    defs = ('<defs>' + "".join(
        f'<marker id="{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for i, c in (("hb", "#1f6feb"), ("hg", "#8c959f"), ("hp", "#8250df"), ("hgr", "#4c8c00"))) + '</defs>')

    def node(x, y, w, h, cls, lines):
        out = [f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="5" stroke-width="1.5"/>']
        for i, t in enumerate(lines):
            ty = y + h / 2 + (i - (len(lines) - 1) / 2) * 19 + 5
            out.append(f'<text class="boxink m15" x="{x + w / 2}" y="{ty:g}" text-anchor="middle">{t}</text>')
        return "\n".join(out)

    def chip(cx, cy, lines, w):
        h = 20 * len(lines) + 4
        out = [f'<rect class="chip" x="{cx - w / 2}" y="{cy - h / 2}" width="{w}" height="{h}"/>']
        for i, t in enumerate(lines):
            ty = cy + (i - (len(lines) - 1) / 2) * 20 + 5
            out.append(f'<text class="boxink m14" x="{cx}" y="{ty:g}" text-anchor="middle">{t}</text>')
        return "\n".join(out)

    nodes = {  # x, y, w, h, class, lines
        "PL": (388, 12, 160, 42, "dynamo", ["Dynamo planner"]),
        "OP": (388, 112, 160, 42, "dynamo", ["Dynamo operator"]),
        "C": (62, 112, 156, 42, "client", ["Client / AIPerf"]),
        "F": (8, 192, 266, 60, "dynamo", ["Dynamo frontend", "HTTP, tokenize, KV-aware router"]),
        "P": (232, 290, 272, 60, "gpu", ["vLLM prefill or aggregated worker", "GPU pool or MIG slice"]),
        "D": (276, 450, 186, 46, "gpu", ["vLLM decode worker"]),
    }
    edges = {  # name: (points, stroke class, marker, dash)
        "c-f": ([(140, 154), (140, 190)], "e-blue", "hb", None),
        "f-p": ([(182, 252), (182, 272), (328, 272), (328, 288)], "e-blue", "hb", None),
        "f-d": ([(100, 252), (100, 432), (328, 432), (328, 448)], "e-blue", "hb", None),
        "p-d": ([(368, 350), (368, 448)], "e-kv", "hp", "3 4"),
        "d-f": ([(276, 473), (50, 473), (50, 254)], "e-green", "hgr", None),
        "f-c": ([(96, 192), (96, 156)], "e-green", "hgr", None),
        "pl-op": ([(468, 54), (468, 110)], "line e-grey", "hg", None),
        "op-p": ([(444, 154), (444, 172), (408, 172), (408, 288)], "line e-grey", "hg", None),
        "op-d": ([(492, 154), (492, 172), (524, 172), (524, 432), (408, 432), (408, 448)], "line e-grey", "hg", None),
    }
    body = [defs]
    for name, (pts, cls, mk, dash) in edges.items():
        width = 2.5 if name == "p-d" else (2 if cls in ("e-blue", "e-green") else 1.5)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<path class="{cls}" d="{rounded(pts)}" stroke-width="{width}"{d} marker-end="url(#{mk})"/>')
    body += [node(*v) for v in nodes.values()]

    # Highlights: the planner sets replica counts, the operator creates both worker pods, then one request runs:
    # client -> frontend -> prefill leg -> decode leg -> the decode worker pulls the KV cache and generates.
    highlights = [("PL", "dynamo", 1, 12), ("OP", "dynamo", 12, 22), ("P", "gpu", 22, 30), ("D", "gpu", 22, 30),
                  ("C", "client", 32, 40), ("F", "dynamo", 40, 48), ("P", "prefill", 50, 60),
                  ("D", "decode", 74, 92), ("C", "client", 86, 98)]
    for i, (key, cls, a, b) in enumerate(highlights):
        x, y, w, h = nodes[key][:4]
        name = f"hl{i}"
        body.append(f'<rect class="{cls} motion" x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
                    f'style="animation: {name} {CYC}s linear infinite; opacity: 0; fill: none; stroke-width: 5px;"/>')
        css += pulse(name, a, b)
    dots = [("d-pl", edges["pl-op"][0], 4, 12, "c-blue", 6), ("d-op1", edges["op-p"][0], 13, 22, "c-blue", 6),
            ("d-op2", edges["op-d"][0], 13, 22, "c-blue", 6), ("d-c", edges["c-f"][0], 33, 40, "c-blue", 7),
            ("d-fp", edges["f-p"][0], 42, 50, "c-blue", 7), ("d-fd", edges["f-d"][0], 58, 66, "c-blue", 7)]
    for i in range(5):  # output tokens: decode worker -> frontend, then frontend -> client
        s0 = 76 + i * 3
        dots += [(f"tk{i}a", edges["d-f"][0], s0, s0 + 6, "c-green", 6),
                 (f"tk{i}b", edges["f-c"][0], s0 + 6.5, s0 + 8, "c-green", 6)]
    for name, pts, a, b, cls, r in dots:
        body.append(f'<circle class="{cls} motion" r="{r}" cx="0" cy="0" '
                    f'style="animation: {name} {CYC}s linear infinite; opacity: 0;"/>')
        css += path_anim(name, pts, a, b)
    for i in range(5):  # KV blocks pulled down the purple edge
        name, s0 = f"kv{i}", 64 + i * 2
        body.append(f'<rect class="c-purple motion" x="-6" y="-6" width="12" height="12" rx="2" '
                    f'style="animation: {name} {CYC}s linear infinite; opacity: 0;"/>')
        css += path_anim(name, [(368, 352), (368, 444)], s0, s0 + 6)
    # Edge labels last, so moving dots pass behind them as the edges do; the KV label sits beside its line.
    body += [chip(468, 82, ["replica counts"], 112),
             chip(408, 222, ["creates pods from the", "DynamoGraphDeployment"], 190),
             chip(262, 391, ["KV cache via NIXL / UCX", "over NVLink (measured),", "RoCE or InfiniBand"], 176),
             chip(163, 473, ["output tokens"], 104)]
    return svg(W, H, "\n".join(body), css)


if __name__ == "__main__":
    for name, fn in (("anim-agg-vs-disagg", agg_vs_disagg), ("anim-interference", interference),
                     ("anim-kv-transfer", kv_transfer), ("anim-request-journey", request_journey),
                     ("anim-stack-overview", stack_overview)):
        p = OUT / f"{name}.svg"
        p.write_text(fn(), encoding='utf-8')
        print("wrote", p.relative_to(OUT.parent), f"{p.stat().st_size / 1024:.1f} KB")
