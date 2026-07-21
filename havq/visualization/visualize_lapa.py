"""
havq/visualization/visualize_lapa.py
====================================
Interactive HTML view of a video's LAPA latent-action labels, for eyeballing
whether the tokens mean anything.

Reads the cached latent actions written by havq/data/preprocessing/lapa_tokenize.py
(<features_dir>/lapa_clip<g>_stride<g>/<video_id>_emb.npy + _meta.json), clusters
them into K action labels at several K, and writes ONE self-contained HTML file:

    <out_dir>/<video_id>_lapa.html

The page references the source .mp4 by relative path and seeks it -- no clip
export step. Clicking any segment plays exactly that time range.

Two views:
  Timeline  -- the video as proportional colored blocks (runs of one label),
               above a change-score chart (1 - cos between adjacent latent
               actions) with segment boundaries marked.
  By label  -- one row per label, gathering every segment carrying it, so you can
               see at a glance whether "label 3" is consistently the same action.
               This is the view that tells you if the embedding is meaningful;
               the timeline alone can't.

K is switchable in-page (labels for every K are precomputed and embedded), so you
can watch segments coarsen without re-running anything.

Latent actions are mean-centered before clustering. That is required, not
cosmetic: the raw deltas carry a constant offset ~90% of a typical row norm, so
uncentered distances mostly measure that offset. Centering uses the video's own
mean, so nothing here needs a corpus.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# same palette as stepsegmenter's visualization, so the two are readable side by side
PALETTE = [
    "#4878d0", "#ee854a", "#6acc65", "#d65f5f", "#956cb4",
    "#8c613c", "#dc7ec0", "#797979", "#d5bb67", "#82c6e2",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def load_video_actions(features_dir: str, video_id: str):
    """Cached latent actions + meta for one video. Returns (embeddings, meta)."""
    emb_path = os.path.join(features_dir, video_id + "_emb.npy")
    meta_path = os.path.join(features_dir, video_id + "_meta.json")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"No cached latent actions at {emb_path}. Run run_lapa_tokenize.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return np.load(emb_path).astype(np.float64), meta


def cluster_labels(embeddings: np.ndarray, k_values, pca_dim: int, seed: int) -> dict:
    """{K: labels} for each K. Centered (see module docstring), optionally PCA'd."""
    X = embeddings - embeddings.mean(axis=0)
    if pca_dim and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    labels_by_k = {}
    for k in k_values:
        if k > len(X):
            continue
        labels_by_k[int(k)] = KMeans(
            n_clusters=int(k), n_init=10, random_state=seed
        ).fit_predict(X).astype(int).tolist()
    return labels_by_k


def change_scores(embeddings: np.ndarray) -> list:
    """1 - cos(e_t, e_t+1) on centered actions: how much the action changes per step."""
    X = embeddings - embeddings.mean(axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = X / norms
    return [float(1.0 - v) for v in (unit[:-1] * unit[1:]).sum(axis=1)]


def relative_video_src(out_dir: str, video_path: str) -> str:
    """Path to the .mp4 as the HTML should reference it (relative if possible)."""
    try:
        return os.path.relpath(video_path, out_dir)
    except ValueError:
        # different drive/mount: fall back to an absolute file URL
        return "file://" + os.path.abspath(video_path)


def find_video(videos_dir: str, video_id: str) -> str:
    """Locate <video_id>.mp4 under videos_dir, flat or one level down."""
    flat = os.path.join(videos_dir, video_id + ".mp4")
    if os.path.exists(flat):
        return flat
    nested = os.path.join(videos_dir, video_id.split("-")[0], video_id + ".mp4")
    if os.path.exists(nested):
        return nested
    raise FileNotFoundError(f"Could not find {video_id}.mp4 under {videos_dir}")


PAGE_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; background: #f7f7f9; color: #222; }
header { padding: 14px 20px 10px; background: #fff; border-bottom: 1px solid #e3e3e8; }
h1 { font-size: 17px; margin: 0 0 4px; font-weight: 600; }
.meta { font-size: 12px; color: #666; }
.player-wrap { position: sticky; top: 0; z-index: 20; background: #fff;
               border-bottom: 1px solid #e3e3e8; padding: 10px 20px;
               display: flex; gap: 16px; align-items: flex-start; }
video { width: 340px; border-radius: 6px; background: #000; flex-shrink: 0; }
.now { font-size: 13px; color: #444; padding-top: 4px; line-height: 1.7; }
.now b { font-size: 15px; }
.controls { padding: 10px 20px 0; background: #fff; }
.ctrl-row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px;
            font-size: 12px; color: #666; flex-wrap: wrap; }
.kbtn { padding: 5px 13px; border: 1px solid #d5d5dd; background: #fff;
        border-radius: 5px; cursor: pointer; font-size: 12px; }
.kbtn.active { background: #4878d0; color: #fff; border-color: #4878d0; }
.mbtn { padding: 5px 11px; border: 1px solid #d5d5dd; background: #fff;
        border-radius: 5px; cursor: pointer; font-size: 12px; }
.mbtn.active { background: #6acc65; color: #fff; border-color: #6acc65; }
.tabs { display: flex; gap: 2px; padding: 8px 20px 0; background: #fff;
        border-bottom: 1px solid #e3e3e8; }
.tab-btn { padding: 7px 18px; border: none; border-radius: 6px 6px 0 0;
           background: #ececf1; cursor: pointer; font-size: 13px; }
.tab-btn.active { background: #4878d0; color: #fff; }
.pane { display: none; padding: 16px 20px 40px; }
.pane.active { display: block; }
.timeline { display: flex; width: 100%; height: 46px; border-radius: 4px;
            overflow: hidden; border: 1px solid #ddd; }
.seg { cursor: pointer; position: relative; transition: filter .1s; }
.seg:hover { filter: brightness(1.18); }
.seg.playing { outline: 2px solid #111; outline-offset: -2px; z-index: 2; }
.axis { position: relative; height: 16px; margin-top: 3px; font-size: 10px; color: #888; }
.axis span { position: absolute; transform: translateX(-50%); }
.lane-row { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.lane-key { width: 96px; flex-shrink: 0; font-size: 12px; display: flex;
            align-items: center; gap: 6px; }
.sw { width: 13px; height: 13px; border-radius: 3px; flex-shrink: 0; }
.lane { position: relative; flex: 1; height: 26px; background: #ececf1;
        border-radius: 3px; overflow: hidden; }
.lane .seg { position: absolute; height: 100%; }
.lane-n { font-size: 11px; color: #888; width: 74px; flex-shrink: 0; }
.hint { font-size: 12px; color: #777; margin: 0 0 10px; }
h3 { font-size: 13px; margin: 18px 0 7px; font-weight: 600; color: #444; }
"""

PAGE_JS = """
let K = null, MINLEN = 1, PLAYING = null;

// Raw runs of one label. MINLEN > 1 additionally absorbs any run shorter than
// MINLEN tokens into its longer neighbour, repeatedly, then re-merges neighbours
// that now share a label. This is COSMETIC POST-PROCESSING, not part of the
// method -- it exists so the timeline is readable when the raw labels flicker.
// MINLEN = 1 is the honest, unfiltered view and is the default.
function runs(labels, minLen) {
  let out = [];
  let s = 0;
  for (let i = 1; i <= labels.length; i++) {
    if (i === labels.length || labels[i] !== labels[s]) {
      out.push({label: labels[s], start: s, end: i});   // [start, end) in tokens
      s = i;
    }
  }
  if (minLen > 1) {
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < out.length && out.length > 1; i++) {
        if (out[i].end - out[i].start >= minLen) continue;
        const L = i > 0 ? out[i - 1] : null;
        const R = i < out.length - 1 ? out[i + 1] : null;
        let tgt;
        if (!L) { tgt = R; }
        else if (!R) { tgt = L; }
        else { tgt = (L.end - L.start) >= (R.end - R.start) ? L : R; }
        tgt.start = Math.min(tgt.start, out[i].start);
        tgt.end = Math.max(tgt.end, out[i].end);
        out.splice(i, 1);
        changed = true;
        break;
      }
    }
    const merged = [];
    out.forEach(r => {
      const last = merged[merged.length - 1];
      if (last && last.label === r.label && last.end === r.start) { last.end = r.end; }
      else { merged.push({label: r.label, start: r.start, end: r.end}); }
    });
    out = merged;
  }
  return out;
}

function fmt(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return m + ":" + String(s).padStart(2, "0");
}

function play(startSec, endSec, el) {
  const v = document.getElementById("player");
  v.currentTime = startSec;
  v.play().catch(() => {});
  if (PLAYING) PLAYING.classList.remove("playing");
  if (el) { el.classList.add("playing"); PLAYING = el; }
  document.getElementById("now").innerHTML =
    "<b>" + fmt(startSec) + " &ndash; " + fmt(endSec) + "</b><br>" +
    (endSec - startSec).toFixed(1) + "s segment";
  clearInterval(window.stopTimer);
  window.stopTimer = setInterval(() => {
    if (v.currentTime >= endSec) { v.pause(); clearInterval(window.stopTimer); }
  }, 40);
}

function segEl(r, cls) {
  const d = document.createElement("div");
  d.className = "seg" + (cls ? " " + cls : "");
  d.style.background = DATA.palette[r.label % DATA.palette.length];
  const a = r.start * DATA.token_sec, b = r.end * DATA.token_sec;
  d.title = "label " + r.label + "  |  " + fmt(a) + "-" + fmt(b) +
            "  |  " + (b - a).toFixed(1) + "s";
  d.onclick = () => play(a, b, d);
  return d;
}

function render() {
  const labels = DATA.levels[K], rs = runs(labels, MINLEN), total = labels.length;

  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  rs.forEach(r => {
    const d = segEl(r);
    d.style.width = (100 * (r.end - r.start) / total) + "%";
    tl.appendChild(d);
  });

  const ax = document.getElementById("axis");
  ax.innerHTML = "";
  const dur = total * DATA.token_sec;
  const step = Math.max(30, Math.round(dur / 10 / 30) * 30);
  for (let t = 0; t <= dur; t += step) {
    const s = document.createElement("span");
    s.style.left = (100 * t / dur) + "%";
    s.textContent = fmt(t);
    ax.appendChild(s);
  }

  const lanes = document.getElementById("lanes");
  lanes.innerHTML = "";
  const byLabel = {};
  rs.forEach(r => { (byLabel[r.label] = byLabel[r.label] || []).push(r); });
  Object.keys(byLabel).map(Number).sort((a, b) => a - b).forEach(lb => {
    const segs = byLabel[lb];
    const secs = segs.reduce((acc, r) => acc + (r.end - r.start) * DATA.token_sec, 0);
    const row = document.createElement("div");
    row.className = "lane-row";
    row.innerHTML =
      '<div class="lane-key"><span class="sw" style="background:' +
      DATA.palette[lb % DATA.palette.length] + '"></span>label ' + lb + "</div>";
    const lane = document.createElement("div");
    lane.className = "lane";
    segs.forEach(r => {
      const d = segEl(r);
      d.style.left = (100 * r.start / total) + "%";
      d.style.width = Math.max(0.25, 100 * (r.end - r.start) / total) + "%";
      lane.appendChild(d);
    });
    row.appendChild(lane);
    const n = document.createElement("div");
    n.className = "lane-n";
    n.textContent = segs.length + "x, " + Math.round(secs) + "s";
    row.appendChild(n);
    lanes.appendChild(row);
  });

  document.getElementById("segcount").textContent =
    rs.length + " segments, " + (dur / rs.length).toFixed(1) + "s avg";
}

function setK(k, btn) {
  K = k;
  document.querySelectorAll(".kbtn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  render();
}

function setMin(m, btn) {
  MINLEN = m;
  document.querySelectorAll(".mbtn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  render();
}

function showTab(name, btn) {
  document.querySelectorAll(".pane").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("pane-" + name).classList.add("active");
  btn.classList.add("active");
}
"""


def svg_change_chart(scores: list, token_sec: float, width: int = 1100, height: int = 110) -> str:
    """Inline SVG of the per-step change score. Same idiom as stepsegmenter's charts."""
    if not scores:
        return ""
    pad_l, pad_r, pad_t, pad_b = 44, 12, 10, 24
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    vmin, vmax = min(scores), max(scores)
    vrange = (vmax - vmin) or 1.0

    points = " ".join(
        f"{pad_l + i / max(len(scores) - 1, 1) * inner_w:.1f},"
        f"{pad_t + (1 - (v - vmin) / vrange) * inner_h:.1f}"
        for i, v in enumerate(scores)
    )
    ticks = ""
    for frac in (0.0, 0.5, 1.0):
        v = vmin + frac * vrange
        y = pad_t + (1 - frac) * inner_h
        ticks += (
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" font-size="9" '
            f'fill="#888">{v:.2f}</text>'
            f'<line x1="{pad_l - 3}" y1="{y:.1f}" x2="{pad_l}" y2="{y:.1f}" stroke="#bbb"/>'
        )
    return (
        f'<svg width="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'style="display:block">'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + inner_h}" stroke="#ccc"/>'
        f'<line x1="{pad_l}" y1="{pad_t + inner_h}" x2="{pad_l + inner_w}" '
        f'y2="{pad_t + inner_h}" stroke="#ccc"/>'
        f'{ticks}'
        f'<polyline points="{points}" fill="none" stroke="#4878d0" stroke-width="1.2"/>'
        f'</svg>'
    )


def build_page(video_id: str, meta: dict, video_src: str, labels_by_k: dict,
               scores: list, token_sec: float) -> str:
    data = {
        "palette": PALETTE,
        "token_sec": token_sec,
        "levels": {str(k): v for k, v in labels_by_k.items()},
    }
    k_buttons = "".join(
        f'<button class="kbtn" onclick="setK(\'{k}\', this)">{k}</button>'
        for k in sorted(labels_by_k)
    )
    # display-only smoothing: absorb runs shorter than N tokens (1 = off)
    m_buttons = "".join(
        f'<button class="mbtn{" active" if m == 1 else ""}" '
        f'onclick="setMin({m}, this)">'
        f'{"raw" if m == 1 else format(m * token_sec, ".1f") + "s"}</button>'
        for m in (1, 2, 3, 5, 8)
    )
    first_k = sorted(labels_by_k)[0]
    duration = len(next(iter(labels_by_k.values()))) * token_sec

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LAPA actions: {html_lib.escape(video_id)}</title>
<style>{PAGE_CSS}</style></head><body>
<header>
  <h1>LAPA latent actions &mdash; {html_lib.escape(video_id)}</h1>
  <div class="meta">
    {len(scores) + 1} tokens &middot; {token_sec:.2f}s each &middot;
    {duration / 60:.1f} min &middot; {meta.get('fps', '?')} fps &middot;
    clustered on the {meta.get('embedding_dim', '?')}-d centered latent action
  </div>
</header>

<div class="player-wrap">
  <video id="player" controls preload="metadata" src="{html_lib.escape(video_src)}"></video>
  <div class="now" id="now">Click any segment to play it.</div>
</div>

<div class="controls">
  <div class="ctrl-row">
    <span><b>Action labels</b> &mdash; how many kinds of action to sort into:</span>{k_buttons}
    <span id="segcount" style="margin-left:10px"></span>
  </div>
  <div class="ctrl-row">
    <span><b>Hide blips shorter than</b> &mdash; declutter only:</span>{m_buttons}
  </div>
  <div class="ctrl-row" style="margin-top:-2px">
    <span style="color:#999">"raw" is the honest picture; the others merge away
      momentary flickers so the timeline is readable. The labels are identical either
      way &mdash; only how they are drawn changes.</span>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('timeline', this)">Timeline</button>
  <button class="tab-btn" onclick="showTab('bylabel', this)">By label</button>
</div>

<div id="pane-timeline" class="pane active">
  <p class="hint">The video left to right. Each coloured block is a stretch where
     the action label stayed the same &mdash; same colour means the same kind of
     motion. Click a block to play exactly that stretch.</p>
  <div class="timeline" id="timeline"></div>
  <div class="axis" id="axis"></div>
  <h3>Change score &mdash; 1 &minus; cos(action<sub>t</sub>, action<sub>t+1</sub>)</h3>
  <p class="hint">Peaks are moments where the latent action turns over. If the
     labels are meaningful, block edges above should line up with peaks here.</p>
  {svg_change_chart(scores, token_sec)}
</div>

<div id="pane-bylabel" class="pane">
  <p class="hint">One lane per action label, showing every place it occurs in the
     video. <b>This is the view that matters:</b> play three or four blocks from the
     same lane. If they show the same kind of motion, that label is a real action. If
     they look unrelated, it is not.</p>
  <div id="lanes"></div>
</div>

<script>const DATA = {json.dumps(data)};</script>
<script>{PAGE_JS}</script>
<script>
  document.querySelector(".kbtn").classList.add("active");
  K = "{first_k}";
  render();
</script>
</body></html>
"""


def run(cfg: dict) -> None:
    from havq.utils.paths import feature_subdir

    features_dir = os.path.join(
        cfg["paths"]["features_dir"],
        feature_subdir(cfg["model"], cfg["pair_gap"], cfg["pair_gap"]),
    )
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    for video_id in cfg["videos"]:
        embeddings, meta = load_video_actions(features_dir, video_id)
        token_sec = meta["n_clip_frame"] / meta["fps"]

        labels_by_k = cluster_labels(
            embeddings, cfg["k_values"], cfg["pca_dim"], cfg["seed"]
        )
        scores = change_scores(embeddings)
        video_path = find_video(cfg["paths"]["videos_dir"], video_id)
        video_src = relative_video_src(out_dir, video_path)

        page = build_page(video_id, meta, video_src, labels_by_k, scores, token_sec)
        out_path = os.path.join(out_dir, f"{video_id}_lapa_h{cfg['pair_gap']}.html")
        with open(out_path, "w") as f:
            f.write(page)
        logger.info(
            f"{video_id}: {len(embeddings)} tokens, K in {sorted(labels_by_k)} -> {out_path}"
        )

    write_index(out_dir)


def write_index(out_dir: str) -> None:
    """Index over every page in out_dir, so all videos and scales are one click apart.

    Rebuilt from a directory scan rather than from the config, so running a second
    pair_gap does not drop the first one's pages from the index.
    """
    pages = sorted(f for f in os.listdir(out_dir) if f.endswith(".html") and f != "index.html")
    rows = []
    for name in pages:
        stem = name[: -len(".html")]
        video_id, _, gap = stem.rpartition("_lapa_h")
        token_sec = int(gap) / 30.0 if gap.isdigit() else None
        if token_sec is None:
            label = stem
        else:
            label = f"{token_sec:.2f}s tokens"
        rows.append(
            f'<tr><td><a href="{html_lib.escape(name)}">{html_lib.escape(video_id)}</a></td>'
            f'<td class="g">{html_lib.escape(label)}</td></tr>'
        )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>LAPA action visualizations</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 40px auto; max-width: 640px; color: #222; }}
h1 {{ font-size: 19px; margin-bottom: 4px; }}
p {{ color: #666; font-size: 13px; line-height: 1.6; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 18px; }}
td {{ padding: 9px 10px; border-bottom: 1px solid #eee; font-size: 14px; }}
td.g {{ color: #888; font-size: 12px; text-align: right; }}
a {{ color: #4878d0; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
</style></head><body>
<h1>LAPA latent-action visualizations</h1>
<p>Each page: the video as colored action-label segments, click any segment to play it.
   The <b>By label</b> tab is the one that shows whether a label means a consistent
   action. Start with <b>min segment = 2</b> to make the timeline readable, then set it
   back to <b>1</b> to see the raw, unfiltered labels.</p>
<table>{"".join(rows)}</table>
</body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)
    logger.info(f"index -> {os.path.join(out_dir, 'index.html')}")
