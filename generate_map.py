#!/usr/bin/env python3
"""Generate map.html — the map-first public viewer for the mind map.

Parses index.html (the published Twine file), extracts every passage's
name and [[links]], auto-tags each card with topic keywords, computes a
layout (force-directed clusters + alphabetical grid for unlinked
cards), and writes a self-contained map.html.

Viewer features: pan/zoom, hover-highlight neighborhoods, click a card
to read the real Twine page in a side panel, search, a keyword bar that
lights up topic layers, and local drag-to-rearrange (persisted in the
visitor's browser only — the published layout never changes).

Regenerate after editing the Twine file:  python3 generate_map.py
"""

import html as html_mod
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "index.html"
OUT = ROOT / "map.html"

# System passages that aren't content cards.
EXCLUDE = {"StoryInit", "Header", "Footer", "Search", "Delete"}

LINK_RE = re.compile(r"\[\[([^\]\[]+?)\]\]")

# Topic vocabulary for auto-tagging: keyword -> regex matched against
# title + content (case-insensitive).
VOCAB = [
    ("linear algebra", r"matri\w+|eigen\w+|vector space|linear map|determinant|svd|singular value|orthogonal"),
    ("calculus", r"derivative|gradient|integral|differentia\w+|jacobian|hessian|taylor"),
    ("probability", r"probabilit\w+|random variable|distribution|expectation|variance|stochastic"),
    ("statistics", r"statistic\w+|hypothesis|regression|estimator|confidence|p-value|sampling"),
    ("bayesian", r"bayes\w*|prior|posterior|likelihood|mcmc|variational"),
    ("optimization", r"optimi\w+|convex|lagrang\w+|descent|minimi\w+|maximi\w+"),
    ("information theory", r"entropy|information theory|kl divergence|mutual information|compression"),
    ("graph theory", r"graph\w*\b|vertex|vertices|adjacency|spanning tree|shortest path"),
    ("neural networks", r"neural network|perceptron|backprop\w*|activation|weights and biases"),
    ("deep learning", r"deep learning|convolution\w*|transformer|attention|embedding|cnn|rnn|lstm"),
    ("reinforcement learning", r"reinforcement|q-learning|policy gradient|reward|markov decision|bandit"),
    ("nlp", r"language model|nlp|token\w*|text corpus|word2vec|natural language"),
    ("game theory", r"game theory|nash|equilibrium|zero-sum|mechanism design"),
    ("geometry & topology", r"topolog\w+|manifold|geometr\w+|curvature|metric space"),
    ("logic & sets", r"\bset theory|predicate|proposition\w*|proof\b|axiom|boolean"),
    ("algorithms", r"algorithm\w*|complexity|big-o|np-hard|dynamic programming|recursion"),
    ("time series", r"time series|autoregress\w+|forecast\w*|seasonal"),
    ("causality", r"causal\w*|counterfactual|confound\w+|instrumental variable"),
    ("people", r"\bborn\b|professor|laureate|pioneer\w*|his work|her work|career"),
    ("history", r"\bhistory\b|historical|in the \d{4}s|century"),
]


def parse_passages(text):
    passages = []
    for m in re.finditer(
        r'<tw-passagedata([^>]*)>(.*?)</tw-passagedata>', text, re.S
    ):
        attrs, body = m.group(1), m.group(2)
        name = re.search(r'name="([^"]*)"', attrs)
        if not name:
            continue
        name = html_mod.unescape(name.group(1))
        if name in EXCLUDE:
            continue
        content = html_mod.unescape(body)
        links = set()
        for lm in LINK_RE.finditer(content):
            inner = lm.group(1)
            if "|" in inner:
                target = inner.split("|", 1)[1]
            elif "->" in inner:
                target = inner.split("->", 1)[1]
            elif "<-" in inner:
                target = inner.split("<-", 1)[0]
            else:
                target = inner
            target = target.strip()
            if target and target != name:
                links.add(target)
        passages.append({
            "name": name,
            "links": sorted(links),
            "content": content,
        })
    return passages


def tag_keywords(passages):
    compiled = [(kw, re.compile(rx, re.I)) for kw, rx in VOCAB]
    for p in passages:
        hay = p["name"] + "\n" + p["content"]
        scores = []
        for kw, rx in compiled:
            hits = len(rx.findall(hay))
            if hits >= 2:
                scores.append((hits, kw))
        scores.sort(reverse=True)
        p["keywords"] = [kw for _, kw in scores[:4]]


def layout(nodes, edges):
    """Force-directed layout for linked clusters (>=4 cards), packed
    side by side; tiny islands and unlinked cards go to an alphabetical
    grid on the right. Twine's stored positions are an auto-grid strip
    43,000px tall — unusable for a one-view map — so we compute our own."""
    import math
    import random

    rng = random.Random(42)
    by_id = {n["id"]: n for n in nodes}

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    comps = {}
    for i in {a for a, b in edges} | {b for a, b in edges}:
        comps.setdefault(find(i), []).append(i)
    big_comps = sorted((c for c in comps.values() if len(c) >= 4),
                       key=len, reverse=True)
    small = sorted((i for c in comps.values() if len(c) < 4 for i in c),
                   key=str.lower)
    linked_all = {i for c in big_comps for i in c}
    isolated = small + sorted(
        (n["id"] for n in nodes
         if n["id"] not in linked_all and n["id"] not in set(small)),
        key=str.lower)

    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def force_layout(ids):
        pos = {i: [rng.uniform(-400, 400), rng.uniform(-300, 300)] for i in ids}
        idset = set(ids)
        comp_edges = [(a, b) for a, b in edges if a in idset and b in idset]
        deg = {i: len(adj.get(i, set()) & idset) for i in ids}
        K = 120.0
        for it in range(350):
            t = 55.0 * (1 - it / 350)
            disp = {i: [0.0, 0.0] for i in ids}
            for idx, a in enumerate(ids):
                pa = pos[a]
                for b in ids[idx + 1:]:
                    pb = pos[b]
                    dx, dy = pa[0] - pb[0], pa[1] - pb[1]
                    d2 = dx * dx + dy * dy or 0.01
                    d = math.sqrt(d2)
                    f = K * K / d2 * 26
                    fx, fy = dx / d * f, dy / d * f
                    disp[a][0] += fx; disp[a][1] += fy
                    disp[b][0] -= fx; disp[b][1] -= fy
            for a, b in comp_edges:
                pa, pb = pos[a], pos[b]
                dx, dy = pa[0] - pb[0], pa[1] - pb[1]
                d = math.hypot(dx, dy) or 0.1
                f = (d * d) / K / 38
                fx, fy = dx / d * f, dy / d * f
                disp[a][0] -= fx; disp[a][1] -= fy
                disp[b][0] += fx; disp[b][1] += fy
            for i in ids:
                g = 0.03 + 0.004 * deg[i]
                disp[i][0] -= pos[i][0] * g
                disp[i][1] -= pos[i][1] * g
            for i in ids:
                dx, dy = disp[i]
                d = math.hypot(dx, dy) or 0.1
                step = min(d, t)
                pos[i][0] += dx / d * step
                pos[i][1] += dy / d * step
        ids_l = list(ids)
        for _ in range(50):
            moved = False
            for idx, a in enumerate(ids_l):
                for b in ids_l[idx + 1:]:
                    dx = pos[a][0] - pos[b][0]
                    dy = pos[a][1] - pos[b][1]
                    if abs(dx) < 190 and abs(dy) < 46:
                        sx = 6 if dx >= 0 else -6
                        sy = 6 if dy >= 0 else -6
                        pos[a][0] += sx; pos[a][1] += sy
                        pos[b][0] -= sx; pos[b][1] -= sy
                        moved = True
            if not moved:
                break
        min_x = min(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        for i in ids:
            pos[i][0] -= min_x
            pos[i][1] -= min_y
        w = max(p[0] for p in pos.values()) + 240
        h = max(p[1] for p in pos.values()) + 60
        return pos, w, h

    cursor_x, cursor_y, row_h = 0.0, 0.0, 0.0
    MAX_ROW_W = 5200
    for comp in big_comps:
        pos, w, h = force_layout(sorted(comp))
        if cursor_x + w > MAX_ROW_W and cursor_x > 0:
            cursor_x = 0
            cursor_y += row_h + 260
            row_h = 0
        for i, (x, y) in pos.items():
            by_id[i]["x"] = x + cursor_x
            by_id[i]["y"] = y + cursor_y
        cursor_x += w + 320
        row_h = max(row_h, h)

    max_x = max((by_id[i]["x"] for i in linked_all), default=0)

    grid_x0 = max_x + 560
    cols = 10
    col_w, row_hh = 260, 46
    for k, name in enumerate(isolated):
        by_id[name]["x"] = grid_x0 + (k % cols) * col_w
        by_id[name]["y"] = (k // cols) * row_hh
    return nodes


def build(passages):
    names = {p["name"] for p in passages}
    degree = {p["name"]: 0 for p in passages}
    edges = []
    for p in passages:
        for t in p["links"]:
            if t in names and t not in EXCLUDE:
                edges.append([p["name"], t])
                degree[p["name"]] += 1
                degree[t] += 1
    nodes = [
        {"id": p["name"], "x": 0, "y": 0, "d": degree[p["name"]],
         "k": p["keywords"]}
        for p in passages
    ]
    layout(nodes, edges)
    keywords = sorted({kw for p in passages for kw in p["keywords"]})
    return nodes, edges, keywords


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Mind Map — full view</title>
<style>
  :root {
    --bg: #101216;
    --card: #1c2027;
    --card-border: #2e3440;
    --card-hover: #262c36;
    --ink: #e8eaed;
    --ink-muted: #9aa3ad;
    --edge: #39414d;
    --edge-hot: #8ab4f8;
    --accent: #8ab4f8;
    --match: #f8c471;
    --panel-bg: #ffffff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { height: 100%; overflow: hidden; background: var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

  #topbar {
    position: fixed; top: 0; left: 0; right: 0; z-index: 20;
    display: flex; flex-direction: column;
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--card-border);
  }
  #topbar .row1 { display: flex; align-items: center; gap: 14px; padding: 9px 16px 6px; }
  #topbar h1 { font-size: 15px; font-weight: 600; color: var(--ink); white-space: nowrap; }
  #topbar .hint { color: var(--ink-muted); font-size: 12px; margin-left: auto; white-space: nowrap; }
  #search {
    width: 240px; max-width: 34vw; padding: 6px 12px; font-size: 13px;
    background: var(--card); color: var(--ink);
    border: 1px solid var(--card-border); border-radius: 8px; outline: none;
  }
  #search:focus { border-color: var(--accent); }
  .tbtn {
    padding: 6px 12px; font-size: 12.5px; cursor: pointer; white-space: nowrap;
    background: var(--card); color: var(--ink);
    border: 1px solid var(--card-border); border-radius: 8px;
  }
  .tbtn:hover { background: var(--card-hover); }
  #chips {
    display: flex; gap: 6px; padding: 2px 16px 9px; overflow-x: auto;
    scrollbar-width: thin;
  }
  .chip {
    padding: 4px 11px; font-size: 12px; cursor: pointer; white-space: nowrap;
    background: transparent; color: var(--ink-muted);
    border: 1px solid var(--card-border); border-radius: 999px;
  }
  .chip:hover { color: var(--ink); border-color: var(--ink-muted); }
  .chip.on { background: var(--accent); border-color: var(--accent); color: #10131a; font-weight: 600; }
  .chip .n { opacity: 0.65; margin-left: 4px; font-size: 11px; }

  #canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }
  #canvas.dragging { cursor: grabbing; }

  .node rect { fill: var(--card); stroke: var(--card-border); stroke-width: 1; rx: 7; }
  .node text { fill: var(--ink); font-size: 12px; pointer-events: none;
    dominant-baseline: middle; text-anchor: middle; }
  .labels-hidden .node text { display: none; }
  .node { cursor: pointer; }
  .node:hover rect, .node.hot rect { fill: var(--card-hover); stroke: var(--accent); }
  .node.dim { opacity: 0.15; }
  .node.selected rect { stroke: var(--accent); stroke-width: 2; }
  .node.match rect { stroke: var(--match); stroke-width: 2; }
  .node.kw rect { stroke: var(--match); }
  .node.moved rect { stroke-dasharray: 3 2; }

  .edge { stroke: var(--edge); stroke-width: 1.2; fill: none; opacity: 0.55; }
  .edge.hot { stroke: var(--edge-hot); opacity: 0.95; stroke-width: 1.6; }
  .edge.dim { opacity: 0.05; }

  #panel {
    position: fixed; top: 0; right: 0; bottom: 0; width: min(560px, 92vw);
    background: var(--panel-bg); z-index: 30;
    box-shadow: -12px 0 32px rgba(0,0,0,0.45);
    transform: translateX(105%); transition: transform 200ms ease;
    display: flex; flex-direction: column;
  }
  #panel.open { transform: translateX(0); }
  #panel header {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: var(--bg);
    border-bottom: 1px solid var(--card-border);
  }
  #panel header .title { color: var(--ink); font-size: 14px; font-weight: 600;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  #panel header a { color: var(--accent); font-size: 12px; text-decoration: none; white-space: nowrap; }
  #panel header button { background: none; border: none; color: var(--ink-muted);
    font-size: 20px; cursor: pointer; padding: 2px 8px; }
  #panel header button:hover { color: var(--ink); }
  #frame { flex: 1; border: 0; width: 100%; background: white; }
  #frame-loading {
    position: absolute; top: 42px; left: 0; right: 0; padding: 10px 14px;
    background: var(--match); color: #10131a; font-size: 12.5px;
    display: none; z-index: 2;
  }
  #panel.loading #frame-loading { display: block; }
</style>
</head>
<body>
<div id="topbar">
  <div class="row1">
    <h1>AI Mind Map</h1>
    <input id="search" type="search" placeholder="Search cards… ( / )" autocomplete="off">
    <button id="fit" class="tbtn" title="Zoom to fit everything">Fit</button>
    <button id="reset" class="tbtn" title="Undo your local rearranging">Reset layout</button>
    <span class="hint">drag empty space to pan · scroll to zoom · drag a card to rearrange (yours only) · click to read</span>
  </div>
  <div id="chips"></div>
</div>
<svg id="canvas"></svg>
<div id="panel">
  <header>
    <span class="title" id="panel-title"></span>
    <a id="panel-open" href="#" target="_blank" rel="noopener">open full page ↗</a>
    <button id="panel-close" title="Close (Esc)">×</button>
  </header>
  <div id="frame-loading">Loading the knowledge base… first open takes a few seconds</div>
  <iframe id="frame" title="Card content"></iframe>
</div>
<script>
const DATA = __DATA__;
const STORE_KEY = 'mindmap-local-positions-v1';

const svg = document.getElementById('canvas');
const NS = 'http://www.w3.org/2000/svg';
const nodesById = new Map(DATA.nodes.map(n => [n.id, n]));

// Local (per-visitor) rearrangements: applied on load, saved on drag.
let localPos = {};
try { localPos = JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch (e) {}
for (const [id, xy] of Object.entries(localPos)) {
  const n = nodesById.get(id);
  if (n && Array.isArray(xy)) { n.x = xy[0]; n.y = xy[1]; }
}

const CARD_H = 30;
const CHAR_W = 6.6;
const PAD = 14;
function cardW(id) { return Math.max(70, Math.min(240, id.length * CHAR_W + PAD * 2)); }

const world = document.createElementNS(NS, 'g');
svg.appendChild(world);
const edgeLayer = document.createElementNS(NS, 'g');
const nodeLayer = document.createElementNS(NS, 'g');
world.appendChild(edgeLayer);
world.appendChild(nodeLayer);

// edges, indexed by endpoint so drags can redraw them
const edgeEls = [];
const edgesByNode = new Map();
function edgePath(a, b) {
  const na = nodesById.get(a), nb = nodesById.get(b);
  const x1 = na.x + cardW(a) / 2, y1 = na.y + CARD_H / 2;
  const x2 = nb.x + cardW(b) / 2, y2 = nb.y + CARD_H / 2;
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - Math.min(60, Math.hypot(x2 - x1, y2 - y1) * 0.12);
  return `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`;
}
for (const [a, b] of DATA.edges) {
  if (!nodesById.has(a) || !nodesById.has(b)) continue;
  const p = document.createElementNS(NS, 'path');
  p.classList.add('edge');
  p.setAttribute('d', edgePath(a, b));
  edgeLayer.appendChild(p);
  const rec = { el: p, a, b };
  edgeEls.push(rec);
  for (const id of [a, b]) {
    if (!edgesByNode.has(id)) edgesByNode.set(id, []);
    edgesByNode.get(id).push(rec);
  }
}

const nodeEls = new Map();
for (const n of DATA.nodes) {
  const g = document.createElementNS(NS, 'g');
  g.classList.add('node');
  if (localPos[n.id]) g.classList.add('moved');
  g.setAttribute('transform', `translate(${n.x},${n.y})`);
  const w = cardW(n.id);
  const rect = document.createElementNS(NS, 'rect');
  rect.setAttribute('width', w);
  rect.setAttribute('height', CARD_H);
  rect.setAttribute('rx', 7);
  const label = document.createElementNS(NS, 'text');
  label.setAttribute('x', w / 2);
  label.setAttribute('y', CARD_H / 2 + 1);
  label.textContent = n.id.length > 34 ? n.id.slice(0, 32) + '…' : n.id;
  g.appendChild(rect);
  g.appendChild(label);
  g.addEventListener('pointerdown', (e) => startNodeDrag(e, n, g));
  // The click that follows pointerup must not reach the canvas's
  // click-empty-space-to-deselect handler, or the panel closes the
  // instant it opens.
  g.addEventListener('click', (e) => e.stopPropagation());
  g.addEventListener('mouseenter', () => focus(n.id, true));
  g.addEventListener('mouseleave', () => focus(null, false));
  nodeLayer.appendChild(g);
  nodeEls.set(n.id, g);
}

const neighbors = new Map();
for (const { a, b } of edgeEls) {
  if (!neighbors.has(a)) neighbors.set(a, new Set());
  if (!neighbors.has(b)) neighbors.set(b, new Set());
  neighbors.get(a).add(b);
  neighbors.get(b).add(a);
}

// ----- highlight state: keyword layer + hover/selection focus -----
let selected = null;
let activeKeywords = new Set();

function baseState() {
  const kwOn = activeKeywords.size > 0;
  for (const [nid, g] of nodeEls) {
    const n = nodesById.get(nid);
    const hit = kwOn && n.k.some(k => activeKeywords.has(k));
    g.classList.toggle('kw', hit);
    g.classList.toggle('dim', kwOn && !hit);
    g.classList.remove('hot');
  }
  for (const { el, a, b } of edgeEls) {
    el.classList.remove('hot');
    const an = nodesById.get(a), bn = nodesById.get(b);
    const dimmed = kwOn && !(an.k.some(k => activeKeywords.has(k)) && bn.k.some(k => activeKeywords.has(k)));
    el.classList.toggle('dim', dimmed);
  }
}

function focus(id, on) {
  const active = on ? id : selected;
  if (!active) { baseState(); return; }
  const near = neighbors.get(active) || new Set();
  for (const [nid, g] of nodeEls) {
    const isNear = nid === active || near.has(nid);
    g.classList.toggle('dim', !isNear);
    g.classList.toggle('hot', isNear && nid !== active);
  }
  for (const { el, a, b } of edgeEls) {
    const touches = a === active || b === active;
    el.classList.toggle('hot', touches);
    el.classList.toggle('dim', !touches);
  }
}

// ----- keyword chips -----
const chipBar = document.getElementById('chips');
const counts = new Map();
for (const n of DATA.nodes) for (const k of n.k) counts.set(k, (counts.get(k) || 0) + 1);
for (const kw of DATA.keywords) {
  const b = document.createElement('button');
  b.className = 'chip';
  b.innerHTML = `${kw}<span class="n">${counts.get(kw) || 0}</span>`;
  b.addEventListener('click', () => {
    if (activeKeywords.has(kw)) activeKeywords.delete(kw);
    else activeKeywords.add(kw);
    b.classList.toggle('on', activeKeywords.has(kw));
    baseState();
  });
  chipBar.appendChild(b);
}

// ----- pan & zoom -----
let view = { x: 0, y: 0, k: 1 };
function topbarH() { return document.getElementById('topbar').offsetHeight; }
function apply() {
  world.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
  world.classList.toggle('labels-hidden', view.k < 0.3);
}
function zoomToFit() {
  const xs = DATA.nodes.map(n => n.x), ys = DATA.nodes.map(n => n.y);
  const minX = Math.min(...xs) - 60, maxX = Math.max(...xs) + 300;
  const minY = Math.min(...ys) - 60, maxY = Math.max(...ys) + 120;
  const th = topbarH();
  const vw = innerWidth, vh = innerHeight - th;
  const k = Math.min(vw / (maxX - minX), vh / (maxY - minY));
  view = { k, x: (vw - (maxX - minX) * k) / 2 - minX * k, y: th + (vh - (maxY - minY) * k) / 2 - minY * k };
  apply();
}
svg.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = Math.exp(-e.deltaY * 0.0015);
  const k2 = Math.min(3, Math.max(0.03, view.k * factor));
  view.x = e.clientX - (e.clientX - view.x) * (k2 / view.k);
  view.y = e.clientY - (e.clientY - view.y) * (k2 / view.k);
  view.k = k2;
  apply();
}, { passive: false });

let panDrag = null;
svg.addEventListener('pointerdown', (e) => {
  panDrag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
  svg.classList.add('dragging');
  svg.setPointerCapture(e.pointerId);
});
svg.addEventListener('pointermove', (e) => {
  if (!panDrag) return;
  view.x = panDrag.vx + (e.clientX - panDrag.x);
  view.y = panDrag.vy + (e.clientY - panDrag.y);
  apply();
});
svg.addEventListener('pointerup', () => { panDrag = null; svg.classList.remove('dragging'); });
svg.addEventListener('click', () => {
  if (selected) { selected = null; baseState(); closePanel(); }
});
document.getElementById('fit').addEventListener('click', zoomToFit);
document.getElementById('reset').addEventListener('click', () => {
  localStorage.removeItem(STORE_KEY);
  location.reload();
});

// ----- node dragging (local rearrange) + click-to-open -----
function startNodeDrag(e, n, g) {
  e.stopPropagation();
  try { g.setPointerCapture(e.pointerId); } catch (err) {}
  const start = { px: e.clientX, py: e.clientY, x: n.x, y: n.y };
  let moved = false;
  const onMove = (ev) => {
    const dx = (ev.clientX - start.px) / view.k;
    const dy = (ev.clientY - start.py) / view.k;
    if (!moved && Math.hypot(ev.clientX - start.px, ev.clientY - start.py) < 5) return;
    moved = true;
    n.x = start.x + dx;
    n.y = start.y + dy;
    g.setAttribute('transform', `translate(${n.x},${n.y})`);
    for (const rec of edgesByNode.get(n.id) || []) {
      rec.el.setAttribute('d', edgePath(rec.a, rec.b));
    }
  };
  const onUp = () => {
    g.removeEventListener('pointermove', onMove);
    g.removeEventListener('pointerup', onUp);
    if (moved) {
      localPos[n.id] = [Math.round(n.x), Math.round(n.y)];
      g.classList.add('moved');
      try { localStorage.setItem(STORE_KEY, JSON.stringify(localPos)); } catch (err) {}
    } else {
      select(n.id);
    }
  };
  g.addEventListener('pointermove', onMove);
  g.addEventListener('pointerup', onUp);
}

// ----- panel -----
const panel = document.getElementById('panel');
const frame = document.getElementById('frame');
const panelTitle = document.getElementById('panel-title');
const panelOpen = document.getElementById('panel-open');
let frameBooted = false;
function passageUrl(id) { return `index.html?passage=${encodeURIComponent(id)}`; }
function select(id) {
  selected = id;
  for (const [nid, g] of nodeEls) g.classList.toggle('selected', nid === id);
  focus(id, true);
  panelTitle.textContent = id;
  panelOpen.href = passageUrl(id);
  panel.classList.add('open');
  if (frameBooted) {
    try { frame.contentWindow.SugarCube.Engine.play(id); return; } catch (e) {}
    try { frame.contentWindow.Engine.play(id); return; } catch (e) {}
  }
  panel.classList.add('loading');
  frame.src = passageUrl(id);
  frameBooted = true;
}
frame.addEventListener('load', () => panel.classList.remove('loading'));
function closePanel() { panel.classList.remove('open'); }
document.getElementById('panel-close').addEventListener('click', closePanel);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { closePanel(); selected = null; baseState(); }
  if (e.key === '/' && document.activeElement !== searchBox) { e.preventDefault(); searchBox.focus(); }
});

// ----- search -----
const searchBox = document.getElementById('search');
searchBox.addEventListener('input', () => {
  const q = searchBox.value.trim().toLowerCase();
  for (const [nid, g] of nodeEls) {
    g.classList.toggle('match', !!q && nid.toLowerCase().includes(q));
  }
});
searchBox.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const q = searchBox.value.trim().toLowerCase();
  if (!q) return;
  const hit = DATA.nodes.find(n => n.id.toLowerCase().includes(q));
  if (!hit) return;
  view.k = Math.max(view.k, 0.9);
  view.x = innerWidth / 2 - (hit.x + cardW(hit.id) / 2) * view.k;
  view.y = (innerHeight + topbarH()) / 2 - (hit.y + 15) * view.k;
  apply();
  select(hit.id);
});

zoomToFit();
</script>
</body>
</html>
"""


def main():
    text = SOURCE.read_text(encoding="utf-8")
    passages = parse_passages(text)
    tag_keywords(passages)
    nodes, edges, keywords = build(passages)
    data = json.dumps(
        {"nodes": nodes, "edges": edges, "keywords": keywords},
        separators=(",", ":"),
    )
    OUT.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    tagged = sum(1 for n in nodes if n["k"])
    print(f"map.html written: {len(nodes)} cards, {len(edges)} links, "
          f"{len(keywords)} keywords ({tagged} cards tagged)")


if __name__ == "__main__":
    main()
