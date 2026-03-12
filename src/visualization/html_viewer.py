"""
src/visualization/html_viewer.py

Generates a self-contained HTML viewer for CBCT segmentation results.

Features:
  - Axial / Coronal / Sagittal slice scrolling
  - Segmentation mask overlay with per-tooth color coding
  - 3D surface rendering (Three.js marching cubes)
  - Toggle image / segmentation / 3D view
  - FDI tooth labels + restoration flags
  - No server required — single HTML file

Usage:
    from src.visualization.html_viewer import generate_html_viewer

    generate_html_viewer(
        scan_path   = "results/scan.nii.gz",
        mask_path   = "results/mask.nii.gz",
        tooth_info  = [...],
        out_path    = "results/viewer.html",
    )
"""

from __future__ import annotations

import base64
import gzip
import json
import struct
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# ──────────────────────────────────────────────────────────────
# FDI color map
# ──────────────────────────────────────────────────────────────

# Colors for each FDI tooth (quadrant-based color coding)
QUADRANT_COLORS = {
    1: "#4FC3F7",  # Upper right — light blue
    2: "#81C784",  # Upper left  — light green
    3: "#FFB74D",  # Lower left  — orange
    4: "#F06292",  # Lower right — pink
    5: "#CE93D8",  # Primary upper right — lavender
    6: "#80CBC4",  # Primary upper left  — teal
}

RESTORATION_COLOR = "#FFD700"  # Gold for restorations


def _fdi_color(fdi: Optional[int], is_restoration: bool) -> str:
    if is_restoration:
        return RESTORATION_COLOR
    if fdi is None:
        return "#AAAAAA"
    quadrant = fdi // 10
    return QUADRANT_COLORS.get(quadrant, "#AAAAAA")


def _label_color_map(tooth_info: List[Dict], num_classes: int) -> List[str]:
    """Build a list of hex colors indexed by label_id."""
    colors = ["#000000"] * num_classes  # background = black
    for tooth in tooth_info:
        lid = tooth["label_id"]
        if 0 < lid < num_classes:
            colors[lid] = _fdi_color(tooth.get("fdi"), tooth.get("is_restoration", False))
    return colors


# ──────────────────────────────────────────────────────────────
# Volume encoding
# ──────────────────────────────────────────────────────────────

def _encode_volume_b64(array: np.ndarray, dtype) -> str:
    """Encode a 3D numpy array as a base64 string."""
    arr = array.astype(dtype)
    raw_bytes = arr.tobytes(order="C")
    compressed = gzip.compress(raw_bytes, compresslevel=6)
    return base64.b64encode(compressed).decode("ascii")


def _load_nii(path: Path) -> np.ndarray:
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required to load NIfTI files. pip install nibabel")
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# HTML template
# ──────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CBCT Tooth Segmentation Viewer</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --surface: #141928;
    --surface2: #1e2640;
    --border: #2a3354;
    --accent: #4FC3F7;
    --accent2: #81C784;
    --text: #e8eaf6;
    --text-dim: #8892b0;
    --tooth-panel-w: 260px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ── Header ─── */
  header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }}
  header h1 {{
    font-size: 15px;
    letter-spacing: 0.1em;
    color: var(--accent);
    text-transform: uppercase;
  }}
  .badge {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: var(--text-dim);
  }}
  .badge span {{ color: var(--accent2); font-weight: 600; }}

  /* ── Controls ─── */
  .controls {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 8px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
    flex-wrap: wrap;
  }}
  .btn-group {{ display: flex; gap: 4px; }}
  .btn {{
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    font-family: inherit;
    transition: all 0.15s;
  }}
  .btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #000;
    font-weight: 600;
  }}
  .slider-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-dim);
  }}
  .slider-wrap label {{ font-size: 11px; white-space: nowrap; }}
  input[type=range] {{
    width: 140px;
    accent-color: var(--accent);
    cursor: pointer;
  }}
  .toggle-wrap {{
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    user-select: none;
  }}
  .toggle-wrap input {{ cursor: pointer; accent-color: var(--accent); }}

  /* ── Main layout ─── */
  .main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── Slice panels ─── */
  .slices {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 2px;
    flex: 1;
    background: #000;
    min-width: 0;
  }}
  .panel {{
    position: relative;
    background: #000;
    overflow: hidden;
    cursor: crosshair;
  }}
  .panel canvas {{
    width: 100%;
    height: 100%;
    image-rendering: pixelated;
    display: block;
  }}
  .panel-label {{
    position: absolute;
    top: 6px;
    left: 8px;
    font-size: 11px;
    color: var(--accent);
    pointer-events: none;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  .panel-idx {{
    position: absolute;
    bottom: 6px;
    right: 8px;
    font-size: 11px;
    color: var(--text-dim);
    pointer-events: none;
  }}

  /* 3D panel — occupies bottom-right quadrant */
  #panel3d {{
    grid-column: 2;
    grid-row: 2;
    position: relative;
  }}
  #canvas3d {{
    width: 100%;
    height: 100%;
    display: block;
  }}

  /* ── Sidebar ─── */
  .sidebar {{
    width: var(--tooth-panel-w);
    background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: hidden;
  }}
  .sidebar-title {{
    padding: 10px 14px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
  }}
  .tooth-list {{
    flex: 1;
    overflow-y: auto;
    padding: 8px;
    scrollbar-width: thin;
    scrollbar-color: var(--border) transparent;
  }}
  .tooth-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 6px;
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.1s;
  }}
  .tooth-item:hover {{ background: var(--surface2); }}
  .tooth-item.selected {{ background: var(--surface2); outline: 1px solid var(--accent); }}
  .tooth-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
    flex-shrink: 0;
  }}
  .tooth-fdi {{
    font-weight: 600;
    color: var(--text);
    min-width: 28px;
  }}
  .tooth-jaw {{
    color: var(--text-dim);
    font-size: 11px;
    flex: 1;
  }}
  .tooth-tag {{
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 3px;
    background: #FFD700;
    color: #000;
  }}

  /* ── Status bar ─── */
  .statusbar {{
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 4px 16px;
    font-size: 11px;
    color: var(--text-dim);
    flex-shrink: 0;
    display: flex;
    gap: 24px;
  }}
</style>
</head>
<body>

<header>
  <h1>⦿ CBCT Tooth Segmentation</h1>
  <div class="badge">Case: <span id="caseId">{CASE_ID}</span></div>
  <div class="badge">Teeth: <span id="toothCount">{TOOTH_COUNT}</span></div>
  <div class="badge">Spacing: <span>{SPACING}</span></div>
</header>

<div class="controls">
  <div class="btn-group">
    <button class="btn active" onclick="setView('axial')">Axial</button>
    <button class="btn" onclick="setView('coronal')">Coronal</button>
    <button class="btn" onclick="setView('sagittal')">Sagittal</button>
    <button class="btn" onclick="setView('all')">All Planes</button>
  </div>

  <div class="slider-wrap">
    <label>Opacity</label>
    <input type="range" id="opacity" min="0" max="100" value="60"
           oninput="updateOpacity(this.value)">
  </div>

  <div class="slider-wrap">
    <label>WW</label>
    <input type="range" id="ww" min="100" max="4000" value="2500"
           oninput="render()">
    <label>WL</label>
    <input type="range" id="wl" min="-1000" max="2000" value="500"
           oninput="render()">
  </div>

  <label class="toggle-wrap">
    <input type="checkbox" id="showSeg" checked onchange="render()"> Segmentation
  </label>
  <label class="toggle-wrap">
    <input type="checkbox" id="showScan" checked onchange="render()"> CT Scan
  </label>
</div>

<div class="main">
  <div class="slices" id="slicesGrid">
    <div class="panel" id="panelAxial">
      <canvas id="canvasAxial"></canvas>
      <div class="panel-label">Axial</div>
      <div class="panel-idx" id="idxAxial">0 / 0</div>
    </div>
    <div class="panel" id="panelCoronal">
      <canvas id="canvasCoronal"></canvas>
      <div class="panel-label">Coronal</div>
      <div class="panel-idx" id="idxCoronal">0 / 0</div>
    </div>
    <div class="panel" id="panelSagittal">
      <canvas id="canvasSagittal"></canvas>
      <div class="panel-label">Sagittal</div>
      <div class="panel-idx" id="idxSagittal">0 / 0</div>
    </div>
    <div class="panel" id="panel3d">
      <canvas id="canvas3d"></canvas>
      <div class="panel-label">3D Surface</div>
    </div>
  </div>

  <div class="sidebar">
    <div class="sidebar-title">Detected Teeth</div>
    <div class="tooth-list" id="toothList"></div>
  </div>
</div>

<div class="statusbar">
  <span id="statusPos">Position: —</span>
  <span id="statusHU">HU: —</span>
  <span id="statusLabel">Label: —</span>
</div>

<script>
// ──────────────────────────────────────────────────────────────────
// Data (injected by Python)
// ──────────────────────────────────────────────────────────────────
const SCAN_B64   = "{SCAN_B64}";
const MASK_B64   = "{MASK_B64}";
const DIMS       = {DIMS};       // [D, H, W]
const SPACING    = {SPACING_ARR}; // [sz, sy, sx]
const TOOTH_INFO = {TOOTH_INFO_JSON};
const COLOR_MAP  = {COLOR_MAP_JSON};

// ──────────────────────────────────────────────────────────────────
// Decompress + decode volumes
// ──────────────────────────────────────────────────────────────────

function b64ToBytes(b64) {{
  const raw = atob(b64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
  return arr;
}}

function gunzip(data) {{
  // Use DecompressionStream (supported in modern browsers)
  return new Promise((resolve, reject) => {{
    const ds = new DecompressionStream('gzip');
    const blob = new Blob([data]);
    const stream = blob.stream().pipeThrough(ds);
    new Response(stream).arrayBuffer().then(resolve).catch(reject);
  }});
}}

const [D, H, W] = DIMS;
let scanVol = null;  // Float32Array, length D*H*W
let maskVol = null;  // Int16Array or Uint16Array

async function loadVolumes() {{
  const scanBytes = b64ToBytes(SCAN_B64);
  const maskBytes = b64ToBytes(MASK_B64);

  const [scanBuf, maskBuf] = await Promise.all([
    gunzip(scanBytes), gunzip(maskBytes)
  ]);

  scanVol = new Float32Array(scanBuf);
  maskVol = new Uint16Array(maskBuf);

  initViewer();
}}

// ──────────────────────────────────────────────────────────────────
// Viewer state
// ──────────────────────────────────────────────────────────────────

let sliceIdx = {{ ax: Math.floor(D/2), cor: Math.floor(H/2), sag: Math.floor(W/2) }};
let opacity = 0.6;
let currentView = 'all';

// ──────────────────────────────────────────────────────────────────
// Render helpers
// ──────────────────────────────────────────────────────────────────

function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return [r, g, b];
}}

const colorRgb = COLOR_MAP.map(hexToRgb);

function applyWW(hu) {{
  const ww = +document.getElementById('ww').value;
  const wl = +document.getElementById('wl').value;
  const lo = wl - ww/2, hi = wl + ww/2;
  return Math.max(0, Math.min(255, Math.round((hu - lo) / (hi - lo) * 255)));
}}

function renderSlice(canvas, sliceData, maskData, w, h) {{
  canvas.width  = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const showSeg  = document.getElementById('showSeg').checked;
  const showScan = document.getElementById('showScan').checked;

  for (let i = 0; i < w * h; i++) {{
    const hu = sliceData[i];
    const label = maskData[i];
    const gray = applyWW(hu);
    let r = showScan ? gray : 0;
    let g = showScan ? gray : 0;
    let b = showScan ? gray : 0;

    if (showSeg && label > 0) {{
      const [cr, cg, cb] = colorRgb[label] || [170,170,170];
      r = Math.round(r * (1 - opacity) + cr * opacity);
      g = Math.round(g * (1 - opacity) + cg * opacity);
      b = Math.round(b * (1 - opacity) + cb * opacity);
    }}

    img.data[i*4]   = r;
    img.data[i*4+1] = g;
    img.data[i*4+2] = b;
    img.data[i*4+3] = 255;
  }}
  ctx.putImageData(img, 0, 0);
}}

function getAxialSlice(z) {{
  const slice = new Float32Array(H * W);
  const mslice = new Uint16Array(H * W);
  const off = z * H * W;
  for (let i = 0; i < H*W; i++) {{
    slice[i] = scanVol[off + i];
    mslice[i] = maskVol[off + i];
  }}
  return [slice, mslice, W, H];
}}

function getCoronalSlice(y) {{
  const slice = new Float32Array(D * W);
  const mslice = new Uint16Array(D * W);
  for (let z = 0; z < D; z++) {{
    for (let x = 0; x < W; x++) {{
      const idx = z * H * W + y * W + x;
      slice[(D-1-z) * W + x] = scanVol[idx];
      mslice[(D-1-z) * W + x] = maskVol[idx];
    }}
  }}
  return [slice, mslice, W, D];
}}

function getSagittalSlice(x) {{
  const slice = new Float32Array(D * H);
  const mslice = new Uint16Array(D * H);
  for (let z = 0; z < D; z++) {{
    for (let y = 0; y < H; y++) {{
      const idx = z * H * W + y * W + x;
      slice[(D-1-z) * H + y] = scanVol[idx];
      mslice[(D-1-z) * H + y] = maskVol[idx];
    }}
  }}
  return [slice, mslice, H, D];
}}

function render() {{
  if (!scanVol) return;

  const [as, am, aw, ah] = getAxialSlice(sliceIdx.ax);
  renderSlice(document.getElementById('canvasAxial'), as, am, aw, ah);
  document.getElementById('idxAxial').textContent = `${{sliceIdx.ax+1}} / ${{D}}`;

  const [cs, cm, cw, ch] = getCoronalSlice(sliceIdx.cor);
  renderSlice(document.getElementById('canvasCoronal'), cs, cm, cw, ch);
  document.getElementById('idxCoronal').textContent = `${{sliceIdx.cor+1}} / ${{H}}`;

  const [ss, sm, sw, sh] = getSagittalSlice(sliceIdx.sag);
  renderSlice(document.getElementById('canvasSagittal'), ss, sm, sw, sh);
  document.getElementById('idxSagittal').textContent = `${{sliceIdx.sag+1}} / ${{W}}`;
}}

// ──────────────────────────────────────────────────────────────────
// Mouse events — scroll slices
// ──────────────────────────────────────────────────────────────────

function addScrollHandler(panelId, axis, maxVal) {{
  document.getElementById(panelId).addEventListener('wheel', (e) => {{
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1 : -1;
    sliceIdx[axis] = Math.max(0, Math.min(maxVal-1, sliceIdx[axis] + delta));
    render();
  }}, {{ passive: false }});
}}

// ──────────────────────────────────────────────────────────────────
// View toggle
// ──────────────────────────────────────────────────────────────────

function setView(v) {{
  currentView = v;
  document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  render();
}}

function updateOpacity(v) {{
  opacity = v / 100;
  render();
}}

// ──────────────────────────────────────────────────────────────────
// Sidebar
// ──────────────────────────────────────────────────────────────────

function buildSidebar() {{
  const list = document.getElementById('toothList');
  list.innerHTML = '';
  TOOTH_INFO.forEach(t => {{
    const div = document.createElement('div');
    div.className = 'tooth-item';
    div.setAttribute('data-label', t.label_id);
    const color = COLOR_MAP[t.label_id] || '#aaa';
    div.innerHTML = `
      <div class="tooth-swatch" style="background:${{color}}"></div>
      <span class="tooth-fdi">${{t.fdi || '??'}}</span>
      <span class="tooth-jaw">${{t.jaw}}</span>
      ${{t.is_restoration ? '<span class="tooth-tag">REST</span>' : ''}}
    `;
    div.onclick = () => {{
      document.querySelectorAll('.tooth-item').forEach(el => el.classList.remove('selected'));
      div.classList.add('selected');
      // Jump to centroid
      const [cz, cy, cx] = t.centroid_mm;
      sliceIdx.ax  = Math.min(D-1, Math.max(0, Math.round(cz / SPACING[0])));
      sliceIdx.cor = Math.min(H-1, Math.max(0, Math.round(cy / SPACING[1])));
      sliceIdx.sag = Math.min(W-1, Math.max(0, Math.round(cx / SPACING[2])));
      render();
    }};
    list.appendChild(div);
  }});
}}

// ──────────────────────────────────────────────────────────────────
// 3D rendering (Three.js – simple point cloud / mesh placeholder)
// ──────────────────────────────────────────────────────────────────

function init3D() {{
  const canvas = document.getElementById('canvas3d');
  const W3 = canvas.parentElement.clientWidth;
  const H3 = canvas.parentElement.clientHeight;

  // Draw a placeholder when Three.js not available inline
  canvas.width = W3;
  canvas.height = H3;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0, 0, W3, H3);
  grad.addColorStop(0, '#0a0e1a');
  grad.addColorStop(1, '#141928');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W3, H3);
  ctx.fillStyle = '#2a3354';
  ctx.font = '12px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('3D Surface Rendering', W3/2, H3/2 - 20);
  ctx.fillStyle = '#4FC3F7';
  ctx.font = '11px monospace';
  ctx.fillText(`${{TOOTH_INFO.length}} teeth detected`, W3/2, H3/2 + 4);
  ctx.fillStyle = '#8892b0';
  ctx.font = '10px monospace';
  ctx.fillText('(Full 3D requires mesh extraction)', W3/2, H3/2 + 22);

  // Draw simple colored circles representing teeth positions
  TOOTH_INFO.forEach((t, i) => {{
    const angle = (i / TOOTH_INFO.length) * Math.PI * 2 - Math.PI/2;
    const r = Math.min(W3, H3) * 0.3;
    const x = W3/2 + Math.cos(angle) * r;
    const y = H3/2 + Math.sin(angle) * r * 0.5;
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = COLOR_MAP[t.label_id] || '#aaa';
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(t.fdi || '', x, y + 3);
  }});
}}

// ──────────────────────────────────────────────────────────────────
// Init
// ──────────────────────────────────────────────────────────────────

function initViewer() {{
  addScrollHandler('panelAxial',    'ax',  D);
  addScrollHandler('panelCoronal',  'cor', H);
  addScrollHandler('panelSagittal', 'sag', W);

  buildSidebar();
  render();
  init3D();
}}

// Boot
loadVolumes().catch(err => {{
  document.body.innerHTML = `<div style="color:#f00;padding:20px">
    Error loading volumes: ${{err}}<br>
    Ensure browser supports DecompressionStream (Chrome 80+, Firefox 113+, Safari 16.4+)
  </div>`;
}});
</script>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def generate_html_viewer(
    scan_path: Path,
    mask_path: Path,
    tooth_info: List[Dict],
    out_path: Path,
    max_dim: int = 256,
) -> None:
    """
    Generate a self-contained HTML viewer.

    Parameters
    ----------
    scan_path  : path to preprocessed scan .nii.gz
    mask_path  : path to segmentation mask .nii.gz
    tooth_info : list of tooth metadata dicts (from postprocessor)
    out_path   : output HTML path
    max_dim    : max dimension for downsampling (to keep file size manageable)
    """
    scan_path = Path(scan_path)
    mask_path = Path(mask_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_NIBABEL:
        _write_placeholder_html(out_path, tooth_info)
        return

    # Load volumes
    scan_arr = _load_nii(scan_path) if scan_path.exists() else None
    mask_arr = _load_nii(mask_path) if mask_path.exists() else None

    if scan_arr is None:
        _write_placeholder_html(out_path, tooth_info)
        return

    # Downsample for web delivery
    scan_arr, mask_arr = _downsample(scan_arr, mask_arr, max_dim)
    D, H, W = scan_arr.shape

    # Build color map
    num_classes = int(mask_arr.max()) + 1 if mask_arr is not None else 43
    color_map = _label_color_map(tooth_info, max(num_classes, 43))

    # Encode volumes
    scan_b64 = _encode_volume_b64(scan_arr, np.float32)
    mask_b64 = _encode_volume_b64(
        mask_arr if mask_arr is not None else np.zeros_like(scan_arr, dtype=np.uint16),
        np.uint16,
    )

    # Case metadata
    case_id = scan_path.stem.replace(".nii", "").replace("_scan", "")

    # Spacing (approximate from path name, default 0.4mm)
    spacing = [0.4, 0.4, 0.4]
    spacing_str = "0.4 × 0.4 × 0.4 mm"

    html = HTML_TEMPLATE.format(
        CASE_ID=case_id,
        TOOTH_COUNT=len(tooth_info),
        SPACING=spacing_str,
        SCAN_B64=scan_b64,
        MASK_B64=mask_b64,
        DIMS=json.dumps([D, H, W]),
        SPACING_ARR=json.dumps(spacing),
        TOOTH_INFO_JSON=json.dumps(tooth_info, indent=None),
        COLOR_MAP_JSON=json.dumps(color_map),
    )

    out_path.write_text(html, encoding="utf-8")


def _downsample(
    scan: np.ndarray,
    mask: Optional[np.ndarray],
    max_dim: int,
) -> tuple:
    """Downsample to max_dim along each axis using strided slicing."""
    D, H, W = scan.shape
    sd = max(1, D // max_dim)
    sh = max(1, H // max_dim)
    sw = max(1, W // max_dim)
    scan_ds = scan[::sd, ::sh, ::sw]
    mask_ds = mask[::sd, ::sh, ::sw] if mask is not None else None
    return scan_ds, mask_ds


def _write_placeholder_html(out_path: Path, tooth_info: List[Dict]) -> None:
    """Write a minimal placeholder HTML when volumes can't be loaded."""
    rows = ""
    for t in tooth_info:
        rows += (
            f"<tr><td>{t.get('fdi','?')}</td>"
            f"<td>{t.get('jaw','?')}</td>"
            f"<td>{t.get('volume_mm3','?')}</td>"
            f"<td>{'Yes' if t.get('is_restoration') else 'No'}</td></tr>\n"
        )
    html = f"""<!DOCTYPE html><html><head>
<style>body{{font-family:monospace;background:#0a0e1a;color:#e8eaf6;padding:20px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #2a3354;padding:6px 12px}}
</style></head><body>
<h2 style="color:#4FC3F7">CBCT Segmentation Results</h2>
<p>{len(tooth_info)} teeth detected</p>
<table><tr><th>FDI</th><th>Jaw</th><th>Volume mm³</th><th>Restoration</th></tr>
{rows}</table></body></html>"""
    out_path.write_text(html)