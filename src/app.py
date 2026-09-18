"""
Interactive Web Application for Testing Component 1 & 2
Run: python -m src.app
Access via browser at: http://localhost:8000
"""

import base64
import io
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from PIL import Image

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from src.ingestion import DocumentIngestion
from src.ocr_engine import ThaiPerceptionEngine

app = FastAPI(title="Thai Financial Document OCR Playground")

# Singleton instances - unwarping disabled to guarantee pixel-perfect bounding box alignment
ingestor = DocumentIngestion(target_dpi=200)
ocr_engine = ThaiPerceptionEngine(
    device="cpu",
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    limit_side_len=2400
)

ROOT_DIR = Path(__file__).resolve().parent.parent


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Thai Financial Document OCR - Playground</title>
  <style>
    :root {
      --bg: #0b0f19;
      --panel-bg: rgba(18, 26, 43, 0.95);
      --card-bg: rgba(26, 36, 60, 0.6);
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: #3b82f6;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.4);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --highlight: rgba(59, 130, 246, 0.25);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      height: 100vh;
      width: 100vw;
      overflow: hidden; /* Prevent whole-page scrolling! */
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Sarabun', sans-serif;
    }

    body {
      display: flex;
      flex-direction: column;
    }

    /* Top Navigation Header */
    header {
      background: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      padding: 0.5rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
      height: 52px;
      z-index: 50;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .logo {
      width: 32px;
      height: 32px;
      background: linear-gradient(135deg, #3b82f6, #10b981);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.85rem;
      color: white;
    }
    .brand h1 { font-size: 1.05rem; font-weight: 600; }
    .brand p { font-size: 0.75rem; color: var(--text-muted); }

    .badges {
      display: flex;
      gap: 0.5rem;
    }
    .badge {
      font-size: 0.72rem;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      color: var(--text-muted);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .badge.green {
      background: rgba(16, 185, 129, 0.12);
      border-color: rgba(16, 185, 129, 0.3);
      color: #34d399;
    }

    /* Main Container */
    main {
      flex: 1;
      min-height: 0; /* Important for flex scrolling */
      padding: 0.6rem 1rem 0.75rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      overflow: hidden;
    }

    /* Controls Bar */
    .controls-bar {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.5rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
      gap: 1rem;
    }

    .file-input-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .btn {
      background: var(--primary);
      color: white;
      border: none;
      padding: 0.45rem 0.9rem;
      border-radius: 7px;
      font-weight: 500;
      cursor: pointer;
      font-size: 0.82rem;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }
    .btn:hover { background: var(--primary-hover); transform: translateY(-1px); }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
      color: var(--text);
    }
    .btn-secondary:hover { background: rgba(255, 255, 255, 0.12); }
    .btn-icon {
      padding: 0.35rem 0.6rem;
      font-size: 0.8rem;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border);
      border-radius: 6px;
      color: var(--text);
      cursor: pointer;
    }
    .btn-icon:hover { background: rgba(255, 255, 255, 0.12); }

    .checkbox-label {
      font-size: 0.8rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
      user-select: none;
    }

    /* Workspace Split Screen */
    .workspace {
      display: grid;
      grid-template-columns: 55% 45%;
      gap: 0.75rem;
      flex: 1;
      min-height: 0; /* Prevents overflow */
      height: 100%;
      overflow: hidden;
    }

    @media (max-width: 1024px) {
      .workspace { grid-template-columns: 1fr; }
    }

    .panel {
      background: var(--panel-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }

    .panel-header {
      padding: 0.55rem 0.9rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.02);
      flex-shrink: 0;
    }
    .panel-header h2 { font-size: 0.88rem; font-weight: 600; display: flex; align-items: center; gap: 0.4rem; }

    .tool-group {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }

    /* Left Stage: Document Viewer */
    .viewer-stage {
      flex: 1;
      min-height: 0;
      overflow: auto; /* Dedicated scroll inside viewer */
      padding: 1.25rem;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      background: #060913;
      position: relative;
    }

    .canvas-container {
      position: relative;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.6);
      border-radius: 4px;
      overflow: hidden;
      display: inline-block;
      transform-origin: top center;
      transition: transform 0.15s ease-out;
    }
    .canvas-container img {
      display: block;
      max-width: 100%;
      height: auto;
    }

    .bbox-overlay {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      pointer-events: none;
    }
    .bbox {
      position: absolute;
      border: 1.5px solid rgba(59, 130, 246, 0.7);
      background: rgba(59, 130, 246, 0.12);
      border-radius: 2px;
      pointer-events: auto;
      cursor: pointer;
      transition: all 0.1s ease;
    }
    .bbox:hover, .bbox.hovered {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.35);
      box-shadow: 0 0 10px var(--accent-glow);
      z-index: 20;
    }
    .bbox.selected {
      border-color: #f59e0b;
      background: rgba(245, 158, 11, 0.4);
      box-shadow: 0 0 12px rgba(245, 158, 11, 0.6);
      z-index: 30;
    }

    /* Right Stage: Debug Inspector Tabs */
    .tabs {
      display: flex;
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      background: rgba(0, 0, 0, 0.25);
    }
    .tab-btn {
      flex: 1;
      background: none;
      border: none;
      padding: 0.6rem 0.8rem;
      color: var(--text-muted);
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.15s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
    }
    .tab-btn:hover { color: var(--text); background: rgba(255, 255, 255, 0.02); }
    .tab-btn.active {
      color: var(--text);
      border-bottom-color: var(--primary);
      background: rgba(255, 255, 255, 0.04);
      font-weight: 600;
    }

    .tab-content {
      flex: 1;
      min-height: 0;
      display: none;
      flex-direction: column;
      overflow: hidden;
    }
    .tab-content.active { display: flex; }

    /* Search & Filter Bar inside Blocks tab */
    .filter-bar {
      padding: 0.5rem 0.8rem;
      border-bottom: 1px solid var(--border);
      background: rgba(0, 0, 0, 0.15);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-shrink: 0;
    }
    .search-input {
      flex: 1;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.35rem 0.6rem;
      color: var(--text);
      font-size: 0.8rem;
      outline: none;
      transition: border-color 0.15s;
    }
    .search-input:focus { border-color: var(--primary); }

    /* Scrollable Blocks List Area */
    .blocks-scroll-container {
      flex: 1;
      min-height: 0;
      overflow-y: auto; /* Crucial independent scroll down! */
      padding: 0.6rem 0.8rem;
    }

    /* Custom Sleek Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.15); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.5); }

    /* Individual Block Item */
    .block-item {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 0.5rem 0.75rem;
      margin-bottom: 0.4rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.12s ease;
      cursor: pointer;
      gap: 0.6rem;
    }
    .block-item:hover, .block-item.hovered {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.12);
    }
    .block-item.selected {
      border-color: #f59e0b;
      background: rgba(245, 158, 11, 0.18);
    }

    .block-left {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      min-width: 0;
      flex: 1;
    }
    .block-idx {
      font-size: 0.7rem;
      font-family: monospace;
      color: var(--text-dim);
      background: rgba(0, 0, 0, 0.3);
      padding: 0.15rem 0.35rem;
      border-radius: 4px;
      flex-shrink: 0;
    }
    .block-text {
      font-size: 0.88rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      color: #f1f5f9;
    }
    .block-right {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      flex-shrink: 0;
    }
    .conf-badge {
      font-size: 0.72rem;
      font-weight: 600;
      padding: 0.15rem 0.45rem;
      border-radius: 5px;
      font-family: monospace;
    }
    .conf-high { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .conf-med { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .conf-low { background: rgba(239, 68, 68, 0.2); color: #f87171; }

    /* Detail Inspector Footer */
    .inspector-card {
      background: rgba(15, 23, 42, 0.9);
      border-top: 1px solid var(--border);
      padding: 0.5rem 0.8rem;
      font-size: 0.78rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-shrink: 0;
    }
    .inspector-meta {
      color: var(--text-muted);
      display: flex;
      gap: 0.8rem;
    }

    /* Markdown & JSON Views */
    .code-scroll-container {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 0.8rem;
    }
    pre {
      background: rgba(0, 0, 0, 0.4);
      padding: 0.8rem;
      border-radius: 7px;
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.8rem;
      color: #cbd5e1;
      white-space: pre-wrap;
      overflow-x: auto;
      line-height: 1.45;
    }

    .spinner {
      border: 2.5px solid rgba(255, 255, 255, 0.1);
      border-top-color: var(--primary);
      border-radius: 50%;
      width: 18px;
      height: 18px;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="logo">TH</div>
      <div>
        <h1>Thai Financial Document OCR Playground</h1>
        <p>Step 4: Information Extraction Perception Layer (th_PP-OCRv5)</p>
      </div>
    </div>
    <div class="badges">
      <span class="badge green">● Model: th_PP-OCRv5_mobile_rec</span>
      <span class="badge">Privacy 100% On-Premises</span>
    </div>
  </header>

  <main>
    <!-- Top Compact Controls Toolbar -->
    <div class="controls-bar">
      <div class="file-input-group">
        <input type="file" id="fileInput" accept="image/*,.pdf" style="display: none;">
        <button class="btn" onclick="document.getElementById('fileInput').click()">
          📁 เลือกไฟล์ภาพหรือ PDF
        </button>
        <button class="btn btn-secondary" onclick="loadSampleReceipt()">
          ⚡ ทดสอบบิลตัวอย่าง (Sample)
        </button>
        <label class="checkbox-label">
          <input type="checkbox" id="deskewCheck"> ปรับมุมเอียงอัตโนมัติ (Deskew)
        </label>
      </div>
      <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div class="spinner" id="loadingSpinner"></div>
        <span id="statusText" style="font-size: 0.82rem; color: var(--text-muted);">พร้อมใช้งาน</span>
      </div>
    </div>

    <!-- Dual Workspace: Pinned Left Image & Dedicated Scroll Right Debug -->
    <div class="workspace">
      <!-- Left: Visual Document Stage with Bounding Box Overlay -->
      <div class="panel">
        <div class="panel-header">
          <h2>📄 การแสดงผลเอกสาร & Bounding Boxes</h2>
          <div class="tool-group">
            <label class="checkbox-label" style="margin-right: 0.4rem;">
              <input type="checkbox" id="toggleBboxCheck" checked onchange="toggleBoundingBoxes(this.checked)"> แสดงกรอบ
            </label>
            <button class="btn-icon" onclick="zoomChange(-0.15)" title="Zoom Out">🔍-</button>
            <button class="btn-icon" id="zoomLabel" onclick="zoomReset()" title="Reset Zoom">100%</button>
            <button class="btn-icon" onclick="zoomChange(0.15)" title="Zoom In">🔍+</button>
            <button class="btn-icon" onclick="zoomFit()" title="Fit Width">⤢ Fit</button>
            <button class="btn-icon" onclick="rotateCurrentDoc(90)" title="หมุนเอกสาร 90 องศาตามเข็ม">⟳ หมุน 90°</button>
            <span id="pageInfo" class="badge">0x0 px</span>
          </div>
        </div>
        <div class="viewer-stage" id="viewerStage">
          <div class="canvas-container" id="canvasContainer">
            <img id="docImage" src="" alt="Document Preview" style="display: none;">
            <div class="bbox-overlay" id="bboxOverlay"></div>
          </div>
          <div id="emptyState" style="color: var(--text-muted); font-size: 0.88rem; align-self: center;">
            👈 กรุณาเลือกไฟล์ภาพบิล หรือกดปุ่ม "ทดสอบบิลตัวอย่าง" เพื่อเริ่มต้น
          </div>
        </div>
      </div>

      <!-- Right: Structured Text & LLM Context with Dedicated Scroll Down -->
      <div class="panel">
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab('blocks')">
            <span>ข้อความที่อ่านได้ (Blocks)</span>
            <span class="badge" id="blockCountBadge">0</span>
          </button>
          <button class="tab-btn" onclick="switchTab('markdown')">LLM Context (Markdown)</button>
          <button class="tab-btn" onclick="switchTab('json')">Raw API JSON</button>
        </div>

        <!-- Tab 1: Scrollable Text Blocks List -->
        <div class="tab-content active" id="tabBlocks">
          <div class="filter-bar">
            <input type="text" id="filterInput" class="search-input" placeholder="🔍 ค้นหาข้อความใน Blocks..." oninput="filterBlocks(this.value)">
            <span id="filterCount" class="badge" style="display: none;">0 พบ</span>
          </div>

          <div class="blocks-scroll-container" id="blocksList">
            <div style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 2rem;">
              ยังไม่มีข้อมูล
            </div>
          </div>

          <!-- Inspector Footer -->
          <div class="inspector-card" id="inspectorCard">
            <div id="inspectorText" style="color: var(--text); font-weight: 500;">
              ชี้หรือคลิกที่กรอบหรือข้อความเพื่อตรวจสอบ
            </div>
            <div class="inspector-meta" id="inspectorMeta">
              <span>พิกัด: -</span>
            </div>
          </div>
        </div>

        <!-- Tab 2: LLM Markdown Preview -->
        <div class="tab-content" id="tabMarkdown">
          <div class="code-scroll-container">
            <pre id="markdownView">ยังไม่มีข้อมูล</pre>
          </div>
        </div>

        <!-- Tab 3: Raw JSON Output -->
        <div class="tab-content" id="tabJson">
          <div class="code-scroll-container">
            <pre id="jsonView">ยังไม่มีข้อมูล</pre>
          </div>
        </div>
      </div>
    </div>
  </main>

  <script>
    let currentData = null;
    let currentZoom = 1.0;
    let selectedIdx = null;

    document.getElementById('fileInput').addEventListener('change', function(e) {
      if (e.target.files.length > 0) {
        uploadAndProcess(e.target.files[0]);
      }
    });

    async function loadSampleReceipt() {
      setStatus('กำลังโหลดบิลตัวอย่าง...', true);
      try {
        const res = await fetch('/api/sample');
        const data = await res.json();
        displayResults(data);
        setStatus('สกัดข้อความภาษาไทยสำเร็จ!', false);
      } catch (err) {
        setStatus('เกิดข้อผิดพลาด: ' + err.message, false);
      }
    }

    async function uploadAndProcess(file) {
      setStatus('กำลังประมวลผล OCR ภาษาไทย...', true);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('auto_deskew', document.getElementById('deskewCheck').checked);

      try {
        const res = await fetch('/api/extract', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        displayResults(data);
        setStatus('สกัดข้อความภาษาไทยสำเร็จ!', false);
      } catch (err) {
        setStatus('เกิดข้อผิดพลาด: ' + err.message, false);
      }
    }

    function setStatus(text, isLoading) {
      document.getElementById('statusText').innerText = text;
      document.getElementById('loadingSpinner').style.display = isLoading ? 'block' : 'none';
    }

    function displayResults(data) {
      currentData = data;
      selectedIdx = null;
      document.getElementById('emptyState').style.display = 'none';

      const docImage = document.getElementById('docImage');
      docImage.src = 'data:image/png;base64,' + data.image_base64;
      docImage.style.display = 'block';

      document.getElementById('pageInfo').innerText = `${data.width}x${data.height} px`;
      document.getElementById('blockCountBadge').innerText = data.text_blocks.length;

      // Reset zoom
      zoomReset();

      // Render Bounding Boxes & Blocks List
      const overlay = document.getElementById('bboxOverlay');
      overlay.innerHTML = '';

      const blocksList = document.getElementById('blocksList');
      blocksList.innerHTML = '';

      data.text_blocks.forEach((block, idx) => {
        // Create Bounding Box div
        const boxDiv = document.createElement('div');
        boxDiv.className = 'bbox';
        boxDiv.id = `bbox-${idx}`;

        const left = (block.box.x_min / data.width) * 100;
        const top = (block.box.y_min / data.height) * 100;
        const w = ((block.box.x_max - block.box.x_min) / data.width) * 100;
        const h = ((block.box.y_max - block.box.y_min) / data.height) * 100;

        boxDiv.style.left = `${left}%`;
        boxDiv.style.top = `${top}%`;
        boxDiv.style.width = `${w}%`;
        boxDiv.style.height = `${h}%`;
        boxDiv.title = `#${idx + 1}: ${block.text} (${(block.confidence * 100).toFixed(1)}%)`;

        boxDiv.onmouseenter = () => hoverItem(idx, true, false);
        boxDiv.onmouseleave = () => hoverItem(idx, false, false);
        boxDiv.onclick = () => selectItem(idx);

        overlay.appendChild(boxDiv);

        // Create List Item
        const item = document.createElement('div');
        item.className = 'block-item';
        item.id = `item-${idx}`;
        item.dataset.text = block.text.toLowerCase();

        const confPct = (block.confidence * 100).toFixed(1);
        const confClass = block.confidence >= 0.9 ? 'conf-high' : (block.confidence >= 0.75 ? 'conf-med' : 'conf-low');

        item.innerHTML = `
          <div class="block-left">
            <span class="block-idx">#${idx + 1}</span>
            <span class="block-text" title="${block.text}">${block.text}</span>
          </div>
          <div class="block-right">
            <span class="conf-badge ${confClass}">${confPct}%</span>
          </div>
        `;

        item.onmouseenter = () => hoverItem(idx, true, true);
        item.onmouseleave = () => hoverItem(idx, false, true);
        item.onclick = () => selectItem(idx);

        blocksList.appendChild(item);
      });

      // Update Markdown & JSON Views
      document.getElementById('markdownView').innerText = data.llm_markdown;
      document.getElementById('jsonView').innerText = JSON.stringify(data, null, 2);
    }

    /* Bi-directional Hover & Scroll into View */
    function hoverItem(idx, isEnter, fromList) {
      if (selectedIdx !== null && selectedIdx === idx) return;

      const bbox = document.getElementById(`bbox-${idx}`);
      const item = document.getElementById(`item-${idx}`);

      if (isEnter) {
        bbox?.classList.add('hovered');
        item?.classList.add('hovered');
        updateInspector(idx);

        // If hovered from Bbox, automatically scroll the right debug list into view!
        if (!fromList && item) {
          item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      } else {
        bbox?.classList.remove('hovered');
        item?.classList.remove('hovered');
        if (selectedIdx !== null) updateInspector(selectedIdx);
      }
    }

    /* Click-to-Select and Auto-Center in Left Image */
    function selectItem(idx) {
      // Deselect old
      if (selectedIdx !== null) {
        document.getElementById(`bbox-${selectedIdx}`)?.classList.remove('selected');
        document.getElementById(`item-${selectedIdx}`)?.classList.remove('selected');
      }

      if (selectedIdx === idx) {
        selectedIdx = null;
        document.getElementById('inspectorText').innerText = 'ชี้หรือคลิกที่กรอบหรือข้อความเพื่อตรวจสอบ';
        document.getElementById('inspectorMeta').innerText = 'พิกัด: -';
        return;
      }

      selectedIdx = idx;
      const bbox = document.getElementById(`bbox-${idx}`);
      const item = document.getElementById(`item-${idx}`);

      bbox?.classList.add('selected');
      item?.classList.add('selected');

      // Smoothly bring the bounding box into the center of the left viewer stage!
      if (bbox) {
        bbox.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
      }
      if (item) {
        item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

      updateInspector(idx);
    }

    function updateInspector(idx) {
      if (!currentData || !currentData.text_blocks[idx]) return;
      const block = currentData.text_blocks[idx];
      const conf = (block.confidence * 100).toFixed(1);
      document.getElementById('inspectorText').innerHTML = `
        <span style="color: #60a5fa; font-weight: 700;">#${idx + 1}</span> &nbsp;${block.text}
      `;
      document.getElementById('inspectorMeta').innerHTML = `
        <span>ความมั่นใจ: <b style="color: #34d399;">${conf}%</b></span>
        <span>x: [${block.box.x_min}, ${block.box.x_max}] y: [${block.box.y_min}, ${block.box.y_max}]</span>
      `;
    }

    /* Search & Filtering */
    function filterBlocks(query) {
      const q = query.trim().toLowerCase();
      const items = document.querySelectorAll('.block-item');
      let visible = 0;

      items.forEach(item => {
        const match = !q || item.dataset.text.includes(q);
        item.style.display = match ? 'flex' : 'none';
        if (match) visible++;
      });

      const badge = document.getElementById('filterCount');
      if (q) {
        badge.innerText = `${visible} จาก ${items.length}`;
        badge.style.display = 'inline-flex';
      } else {
        badge.style.display = 'none';
      }
    }

    /* Zoom & Bounding Box Controls */
    function zoomChange(delta) {
      currentZoom = Math.min(Math.max(0.3, currentZoom + delta), 2.5);
      applyZoom();
    }

    function zoomReset() {
      currentZoom = 1.0;
      applyZoom();
    }

    function zoomFit() {
      const stage = document.getElementById('viewerStage');
      const img = document.getElementById('docImage');
      if (!img || !img.naturalWidth) return;
      const ratio = (stage.clientWidth - 40) / img.naturalWidth;
      currentZoom = Math.max(0.2, Math.min(ratio, 1.5));
      applyZoom();
    }

    function applyZoom() {
      const container = document.getElementById('canvasContainer');
      container.style.transform = `scale(${currentZoom})`;
      document.getElementById('zoomLabel').innerText = `${Math.round(currentZoom * 100)}%`;
    }

    function toggleBoundingBoxes(show) {
      document.getElementById('bboxOverlay').style.display = show ? 'block' : 'none';
    }

    /* Tabs Switching */
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      if (tabId === 'blocks') {
        document.querySelector('.tab-btn:nth-child(1)').classList.add('active');
        document.getElementById('tabBlocks').classList.add('active');
      } else if (tabId === 'markdown') {
        document.querySelector('.tab-btn:nth-child(2)').classList.add('active');
        document.getElementById('tabMarkdown').classList.add('active');
      } else if (tabId === 'json') {
        document.querySelector('.tab-btn:nth-child(3)').classList.add('active');
        document.getElementById('tabJson').classList.add('active');
      }
    }

    async function rotateCurrentDoc(degrees) {
      const docImg = document.getElementById('docImage');
      if (!currentData || !docImg.src || docImg.style.display === 'none') {
        alert('กรุณาอัปโหลดเอกสารก่อนทำการหมุน');
        return;
      }
      setStatus('กำลังหมุนภาพ 90° และประมวลผล OCR ใหม่...', true);
      const img = new Image();
      img.src = docImg.src;
      await new Promise(resolve => { img.onload = resolve; });

      const canvas = document.createElement('canvas');
      canvas.width = img.height;
      canvas.height = img.width;
      const ctx = canvas.getContext('2d');
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate((degrees * Math.PI) / 180);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);

      canvas.toBlob(blob => {
        const rotatedFile = new File([blob], 'rotated_doc.png', { type: 'image/png' });
        uploadAndProcess(rotatedFile);
      }, 'image/png');
    }
  </script>
</body>
</html>
"""


@app.post("/api/extract")
async def extract_document(
    file: UploadFile = File(...),
    auto_deskew: bool = Form(False)
):
    file_bytes = await file.read()
    pages = ingestor.load_document(file_bytes)

    if not pages:
        return JSONResponse({"error": "Failed to parse document pages"}, status_code=400)

    first_page = pages[0]
    processed_image = first_page.image

    if auto_deskew:
        processed_image = ingestor.deskew_image(processed_image)

    perception = ocr_engine.process_image(processed_image, page_number=1)

    return {
        "filename": file.filename,
        "width": processed_image.width,
        "height": processed_image.height,
        "image_base64": pil_to_base64(processed_image),
        "text_blocks": [block.model_dump() for block in perception.text_blocks],
        "raw_text": perception.raw_text,
        "llm_markdown": perception.to_llm_markdown()
    }


@app.get("/api/sample")
def get_sample():
    sample_path = ROOT_DIR / "tests" / "output" / "sample_receipt.png"
    if not sample_path.exists():
        from tests.test_components_1_2 import create_sample_receipt_image
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        img = create_sample_receipt_image()
        img.save(sample_path)
    else:
        img = Image.open(sample_path)

    perception = ocr_engine.process_image(img, page_number=1)

    return {
        "filename": "sample_receipt.png",
        "width": img.width,
        "height": img.height,
        "image_base64": pil_to_base64(img),
        "text_blocks": [block.model_dump() for block in perception.text_blocks],
        "raw_text": perception.raw_text,
        "llm_markdown": perception.to_llm_markdown()
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Thai Financial Document OCR UI...")
    print("👉 Open your browser at: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
