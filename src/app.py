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
from pydantic import BaseModel
import uvicorn
from PIL import Image

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from src.ingestion import DocumentIngestion
from src.ocr_engine import ThaiPerceptionEngine
from src.llm_extractor import LLMExtractor
from src.validator import FinancialDocumentValidator
from src.pipeline import DocumentProcessingPipeline
from src.benchmark import ModelBenchmarker
from src.schemas import (
    DOCUMENT_TYPE_TITLES,
    DocumentType,
    FullPipelineResult,
    DocumentValidationResult,
    BenchmarkRequest,
    BenchmarkReport,
    HealthCheckResponse,
    TypeDirectedExtractionResult,
)

tags_metadata = [
    {
        "name": "End-to-End Extraction Pipeline",
        "description": "Complete production pipeline integrating Ingestion, OCR, LLM Reasoning, and Normalization/Validation."
    },
    {
        "name": "Data Validation & Rules",
        "description": "Standalone data validation, Thai date normalization, Thai Tax ID Mod 11, and university rules."
    },
    {
        "name": "Model Benchmarking",
        "description": "Comparative evaluation of LLM models for academic thesis and seminar presentations."
    },
    {
        "name": "System Health",
        "description": "Health check and component status monitoring."
    },
    {
        "name": "Interactive Playground",
        "description": "Interactive Web UI and debug inspector endpoints."
    }
]

app = FastAPI(
    title="Thai Financial Document Extraction API (Component 5)",
    description=(
        "Senior Project: Web-based Automated Expense Reimbursement System using AI\n\n"
        "Faculty of Engineering, Mahidol University\n\n"
        "Unified Pipeline: Document Ingestion (C1) -> Thai PP-OCRv5 Perception (C2) -> "
        "Local LLM Reasoning (C3) -> Business Rules & Normalization (C4) -> "
        "Unified REST Service & Benchmarker (C5)."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Singleton instances - unwarping disabled to guarantee pixel-perfect bounding box alignment
ingestor = DocumentIngestion(target_dpi=200)
ocr_engine = ThaiPerceptionEngine(
    device="cpu",
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    limit_side_len=2400
)
llm_extractor = LLMExtractor(default_model="qwen2.5:3b")
validator = FinancialDocumentValidator(petty_cash_threshold=10000.0)

# Component 5: Unified Pipeline & Benchmarker
pipeline = DocumentProcessingPipeline(
    ingestor=ingestor,
    ocr_engine=ocr_engine,
    llm_extractor=llm_extractor,
    validator=validator,
)
benchmarker = ModelBenchmarker(
    extractor=llm_extractor,
    validator=validator,
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

    /* Extraction Tab Styles */
    .extraction-container {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 0.8rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .extract-ctrl-bar {
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.6rem 0.8rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.6rem;
      flex-wrap: wrap;
    }
    .field-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.8rem;
    }
    .field-card-title {
      font-size: 0.82rem;
      font-weight: 600;
      color: #94a3b8;
      margin-bottom: 0.6rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .field-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 0.6rem;
    }
    .field-item {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      background: rgba(0, 0, 0, 0.2);
      padding: 0.45rem 0.6rem;
      border-radius: 6px;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .field-label {
      font-size: 0.7rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .field-value {
      font-size: 0.85rem;
      color: var(--text);
      font-weight: 500;
      word-break: break-word;
    }
    .items-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      margin-top: 0.3rem;
    }
    .items-table th {
      background: rgba(0, 0, 0, 0.35);
      padding: 0.5rem 0.6rem;
      text-align: left;
      color: var(--text-muted);
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      font-size: 0.75rem;
      text-transform: uppercase;
    }
    .items-table td {
      padding: 0.45rem 0.6rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      color: var(--text);
    }
    .items-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }
    .num-col { text-align: right; font-family: monospace; }
    .totals-box {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(59, 130, 246, 0.08));
      border: 1px solid rgba(16, 185, 129, 0.3);
      border-radius: 8px;
      padding: 0.8rem 1rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.8rem;
    }
    .total-amount-display {
      font-size: 1.4rem;
      font-weight: 700;
      color: #34d399;
      font-family: monospace;
    }
    .cot-accordion {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 8px;
      overflow: hidden;
    }
    .cot-header {
      padding: 0.5rem 0.8rem;
      background: rgba(59, 130, 246, 0.12);
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
      color: #60a5fa;
      display: flex;
      justify-content: space-between;
      align-items: center;
      user-select: none;
    }
    .cot-body {
      padding: 0.7rem 0.8rem;
      font-size: 0.78rem;
      color: #cbd5e1;
      line-height: 1.5;
      font-family: monospace;
      white-space: pre-wrap;
      border-top: 1px solid rgba(59, 130, 246, 0.2);
      max-height: 220px;
      overflow-y: auto;
    }

    /* Component 4 Validation Card Styles */
    .val-gate-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 0.5rem;
      margin-top: 0.2rem;
    }
    .val-gate-item {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      padding: 0.55rem 0.7rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .gate-title {
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 600;
    }
    .gate-desc {
      font-size: 0.78rem;
      color: #f1f5f9;
      font-weight: 500;
      line-height: 1.35;
    }
    .val-issue-item {
      padding: 0.45rem 0.65rem;
      border-radius: 6px;
      font-size: 0.76rem;
      display: flex;
      align-items: flex-start;
      gap: 0.45rem;
      line-height: 1.4;
    }
    .val-issue-error {
      background: rgba(239, 68, 68, 0.12);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
    }
    .val-issue-warning {
      background: rgba(245, 158, 11, 0.12);
      border: 1px solid rgba(245, 158, 11, 0.3);
      color: #fde68a;
    }

    /* Pipeline Timing Breakdown (Component 5) */
    .timing-bar {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.35rem 0.6rem;
      font-size: 0.75rem;
      flex-wrap: wrap;
    }

    /* Benchmark Modal (Component 5) */
    .modal-overlay {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(6px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
    }
    .modal-overlay.active {
      display: flex;
    }
    .modal-card {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 12px;
      width: 100%;
      max-width: 860px;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
      overflow: hidden;
    }
    .modal-header {
      padding: 0.9rem 1.25rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(30, 41, 59, 0.7);
    }
    .modal-header h3 {
      font-size: 1rem;
      font-weight: 600;
      color: #f1f5f9;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .modal-body {
      padding: 1.25rem;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .modal-footer {
      padding: 0.75rem 1.25rem;
      border-top: 1px solid var(--border);
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 0.6rem;
      background: rgba(30, 41, 59, 0.7);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="logo">TH</div>
      <div>
        <h1>Thai Financial Document OCR & Extraction Playground</h1>
        <p>Production Pipeline, REST API & Model Benchmarking (Components 1-5)</p>
      </div>
    </div>
    <div class="badges">
      <a href="/docs" target="_blank" class="badge" style="color: #60a5fa; border-color: rgba(96, 165, 250, 0.4); text-decoration: none; font-weight: 500;">📖 Swagger API Docs (/docs)</a>
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
          <button class="tab-btn" onclick="switchTab('extraction')" id="tabBtnExtraction">
            <span>🤖 สกัดข้อมูลการเงิน (AI)</span>
            <span class="badge green" id="extractionBadge" style="display: none;">Ready</span>
          </button>
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

        <!-- Tab 4: Component 3 Financial Extraction Result -->
        <div class="tab-content" id="tabExtraction">
          <div class="extraction-container">
            <!-- Controls Bar -->
            <div class="extract-ctrl-bar">
              <div style="display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap;">
                <label style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600;">📄 ประเภทเอกสาร (Manual Input):</label>
                <select id="llmDocTypeSelect" class="search-input" style="width: auto; min-width: 240px; padding: 0.25rem 0.5rem;"></select>

                <label style="font-size: 0.78rem; color: var(--text-muted); margin-left: 0.3rem;">โมเดล LLM:</label>
                <select id="llmModelSelect" class="search-input" style="width: auto; min-width: 130px; padding: 0.25rem 0.5rem;"></select>

                <label style="font-size: 0.78rem; color: var(--text-muted); margin-left: 0.3rem;">อุณหภูมิ (Temp):</label>
                <select id="llmTempSelect" class="search-input" style="width: auto; min-width: 130px; padding: 0.25rem 0.5rem;">
                  <option value="0.0" selected>0.0 (แม่นยำสูงสุด / Zero Hallucination)</option>
                  <option value="0.1">0.1 (แม่นยำสูง)</option>
                  <option value="0.2">0.2 (ปกติ)</option>
                </select>

                <label class="checkbox-label" style="font-size: 0.78rem;">
                  <input type="checkbox" id="forceMockCheck"> บังคับใช้ Mock Snapshot (Offline Mode)
                </label>
              </div>
              <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span id="ollamaStatusBadge" class="badge">ตรวจสถานะ Ollama...</span>
                <button class="btn btn-secondary" id="btnOpenBenchmark" onclick="openBenchmarkModal()" title="เปรียบเทียบประสิทธิภาพโมเดล LLM เชิงตัวเลข">
                  📊 เปรียบเทียบโมเดล (Benchmark)
                </button>
                <button class="btn" id="btnExtractLLM" onclick="runLLMExtraction()">
                  ⚡ สกัดข้อมูลด้วย AI
                </button>
              </div>
            </div>

            <div id="extractionEmpty" style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 2.5rem;">
              กรุณาเลือกประเภทเอกสารใน Dropdown แล้วกดปุ่ม <b>"⚡ สกัดข้อมูลด้วย AI"</b> เพื่อให้โมเดลสกัดข้อมูลเฉพาะประเภท
            </div>

            <div id="extractionResultsArea" style="display: none; flex-direction: column; gap: 0.75rem;">
              <!-- Meta & Latency Banner -->
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                  <span id="resDocTypeBadge" class="badge green">1. เอกสารขออนุมัติหลักการ</span>
                  <span id="resModelBadge" class="badge">qwen2.5:3b</span>
                  <span id="resModeBadge" class="badge">Local GPU</span>
                  <span id="resLatencyBadge" class="badge">⏱️ - ms</span>
                  <span id="resValStatusBadge" class="badge green" style="display: none;">✓ ตรวจสอบผ่าน</span>
                </div>
                <div style="display: flex; gap: 0.4rem;">
                  <button class="btn btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="copyExtractionJson()">📋 คัดลอก JSON</button>
                </div>
              </div>

              <!-- Pipeline Timing Stages Breakdown (Component 5) -->
              <div class="timing-bar" id="pipelineTimingsBanner" style="display: none;">
                <span style="font-weight: 600; color: #94a3b8;">⏱️ Pipeline Latency:</span>
                <span id="timingOcrTag" class="badge">OCR: - ms</span>
                <span id="timingLlmTag" class="badge">LLM: - ms</span>
                <span id="timingValTag" class="badge">Validation: - ms</span>
                <span id="timingTotalTag" class="badge green" style="font-weight: 600;">Total: - ms</span>
              </div>

              <!-- Collapsible CoT Reasoning -->
              <div class="cot-accordion" id="cotAccordion" style="display: none;">
                <div class="cot-header" onclick="toggleCot()">
                  <span>🧠 ลำดับความคิดวิเคราะห์ (Chain-of-Thought / &lt;think&gt;)</span>
                  <span id="cotToggleIcon">▼</span>
                </div>
                <div class="cot-body" id="cotBody"></div>
              </div>

              <!-- Component 4: Validation & Rules Engine Status Card -->
              <div class="field-card" id="validationResultCard" style="display: none;">
                <div class="field-card-title">
                  <span>🛡️ การตรวจสอบความถูกต้อง & กฎระเบียบ (Component 4 Rules Engine)</span>
                  <span id="valOverallBadge" class="badge green">PASSED</span>
                </div>
                
                <!-- 4 Quality Gates Grid -->
                <div class="val-gate-grid">
                  <div class="val-gate-item" id="gateDateBox">
                    <span class="gate-title">📅 รูปแบบวันที่ (Thai Date Normalization)</span>
                    <span class="gate-desc" id="gateDateDesc">-</span>
                  </div>
                  <div class="val-gate-item" id="gateTaxBox">
                    <span class="gate-title">🏢 เลขประจำตัวผู้เสียภาษี (Tax ID Mod 11)</span>
                    <span class="gate-desc" id="gateTaxDesc">-</span>
                  </div>
                  <div class="val-gate-item" id="gateMathBox">
                    <span class="gate-title">🧮 กระทบยอดตัวเลข (Financial Math Reconciliation)</span>
                    <span class="gate-desc" id="gateMathDesc">-</span>
                  </div>
                  <div class="val-gate-item" id="gatePolicyBox">
                    <span class="gate-title">📜 กฎระเบียบและเงื่อนไข (University Policy)</span>
                    <span class="gate-desc" id="gatePolicyDesc">-</span>
                  </div>
                </div>

                <!-- Issues List (if any) -->
                <div id="valIssuesContainer" style="display: none; margin-top: 0.7rem; flex-direction: column; gap: 0.4rem;">
                  <div style="font-size: 0.76rem; font-weight: 600; color: #cbd5e1;">⚠️ รายการข้อผิดพลาดและข้อสังเกต (Issues):</div>
                  <div id="valIssuesList" style="display: flex; flex-direction: column; gap: 0.35rem;"></div>
                </div>
              </div>

              <!-- Dynamic Document Fields Card -->
              <div class="field-card" id="documentFieldsCard">
                <div class="field-card-title">
                  <span>🏢 ข้อมูลสำคัญของเอกสาร (Extracted Key Fields)</span>
                  <span id="docTypeTag" class="badge green">ระบุประเภท</span>
                </div>
                <div class="field-grid" id="fieldGrid">
                  <!-- Generated dynamically from result.fields -->
                </div>
              </div>

              <!-- Line Items Table Card -->
              <div class="field-card" id="lineItemsCard">
                <div class="field-card-title">
                  <span>🛒 รายการสินค้าและบริการ (Line Items)</span>
                  <span id="valItemCount" class="badge">0 รายการ</span>
                </div>
                <div style="overflow-x: auto;">
                  <table class="items-table">
                    <thead>
                      <tr>
                        <th style="width: 35px;">#</th>
                        <th>รายการ (Description)</th>
                        <th class="num-col" style="width: 70px;">จำนวน</th>
                        <th style="width: 60px;">หน่วย</th>
                        <th class="num-col" style="width: 90px;">ราคา/หน่วย</th>
                        <th class="num-col" style="width: 90px;">ราคารวม (฿)</th>
                      </tr>
                    </thead>
                    <tbody id="lineItemsTableBody"></tbody>
                  </table>
                </div>
              </div>

              <!-- Totals & Financial Reconciliation -->
              <div class="totals-box" id="totalsBoxContainer">
                <div style="display: flex; flex-direction: column; gap: 0.3rem;">
                  <div id="subtotalRow" style="font-size: 0.8rem; color: var(--text-muted); display: none;">
                    รวมเป็นเงิน (Subtotal): <b id="valSubtotal" style="color: var(--text); font-family: monospace;">-</b>
                  </div>
                  <div id="vatRow" style="font-size: 0.8rem; color: var(--text-muted); display: none;">
                    ภาษีมูลค่าเพิ่ม 7% (VAT): <b id="valVat" style="color: var(--text); font-family: monospace;">-</b>
                  </div>
                  <div id="mathValidationBadge" class="badge green" style="align-self: flex-start; margin-top: 0.2rem; display: none;">
                    ✓ ตรวจสอบยอดถูกต้อง
                  </div>
                </div>
                <div style="text-align: right;">
                  <div style="font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase;">ยอดเงินรวมทั้งสิ้น (Total)</div>
                  <div class="total-amount-display" id="valTotal">฿0.00</div>
                </div>
              </div>
            </div>
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
      } else if (tabId === 'extraction') {
        document.getElementById('tabBtnExtraction').classList.add('active');
        document.getElementById('tabExtraction').classList.add('active');
      } else if (tabId === 'json') {
        document.querySelector('.tab-btn:nth-child(4)').classList.add('active');
        document.getElementById('tabJson').classList.add('active');
      }
    }

    /* Component 3: LLM Financial Extraction Logic */
    let lastExtractionResult = null;

    async function loadLLMStatus() {
      try {
        const res = await fetch('/api/llm/status');
        const data = await res.json();
        const badge = document.getElementById('ollamaStatusBadge');
        const select = document.getElementById('llmModelSelect');
        select.innerHTML = '';

        if (data.ollama_online) {
          badge.className = 'badge green';
          badge.innerText = '🟢 Ollama พร้อมใช้งาน (GPU)';
          if (data.available_models && data.available_models.length > 0) {
            data.available_models.forEach(m => {
              const opt = document.createElement('option');
              opt.value = m;
              opt.innerText = m;
              if (m === data.default_model) opt.selected = true;
              select.appendChild(opt);
            });
          } else {
            const opt = document.createElement('option');
            opt.value = 'qwen2.5:3b';
            opt.innerText = 'qwen2.5:3b';
            select.appendChild(opt);
          }
        } else {
          badge.className = 'badge';
          badge.style.color = '#fbbf24';
          badge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
          badge.style.background = 'rgba(245, 158, 11, 0.1)';
          badge.innerText = '🟠 Ollama Offline (ใช้ Real Snapshot)';
          const opt = document.createElement('option');
          opt.value = 'qwen2.5:3b (Mock)';
          opt.innerText = 'qwen2.5:3b (Real Snapshot)';
          select.appendChild(opt);
          document.getElementById('forceMockCheck').checked = true;
        }

        // Populate Document Type Dropdown (and Benchmark modal dropdown)
        const docSelect = document.getElementById('llmDocTypeSelect');
        const bmDocSelect = document.getElementById('bmDocTypeSelect');
        docSelect.innerHTML = '';
        if (bmDocSelect) bmDocSelect.innerHTML = '';
        if (data.supported_document_types && data.supported_document_types.length > 0) {
          data.supported_document_types.forEach((dt, idx) => {
            const opt = document.createElement('option');
            opt.value = dt.id;
            opt.innerText = dt.title;
            if (idx === 0) opt.selected = true;
            docSelect.appendChild(opt);

            if (bmDocSelect) {
              const bmOpt = document.createElement('option');
              bmOpt.value = dt.id;
              bmOpt.innerText = dt.title;
              if (dt.id === 'general_receipt' || idx === 0) bmOpt.selected = true;
              bmDocSelect.appendChild(bmOpt);
            }
          });
        }
      } catch (err) {
        console.warn('Could not load LLM status:', err);
      }
    }

    async function runLLMExtraction() {
      if (!currentData || !currentData.llm_markdown) {
        alert('กรุณาอัปโหลดเอกสารหรือกดปุ่ม "ทดสอบบิลตัวอย่าง" ก่อนสกัดข้อมูลด้วย AI');
        return;
      }

      const btn = document.getElementById('btnExtractLLM');
      const originalText = btn.innerText;
      btn.innerText = '⏳ กำลังประมวลผล...';
      btn.disabled = true;

      const modelName = document.getElementById('llmModelSelect').value;
      const docType = document.getElementById('llmDocTypeSelect').value || 'principle_approval_request';
      const forceMock = document.getElementById('forceMockCheck').checked;
      const tempVal = parseFloat(document.getElementById('llmTempSelect')?.value || '0.0');

      try {
        const res = await fetch('/api/llm/extract', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ocr_markdown: currentData.llm_markdown,
            document_type: docType,
            model_name: modelName.replace(' (Mock)', ''),
            temperature: tempVal,
            force_mock: forceMock
          })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Extraction failed');
        }

        const result = await res.json();
        lastExtractionResult = result;
        renderExtractionResult(result);
        switchTab('extraction');
      } catch (err) {
        alert('เกิดข้อผิดพลาดในการสกัดข้อมูล: ' + err.message);
      } finally {
        btn.innerText = originalText;
        btn.disabled = false;
      }
    }

    function renderExtractionResult(result) {
      document.getElementById('extractionEmpty').style.display = 'none';
      const area = document.getElementById('extractionResultsArea');
      area.style.display = 'flex';

      // Document Type Badges
      const docTypeName = result.document_type_name_th || result.document_type;
      document.getElementById('resDocTypeBadge').innerText = docTypeName;
      document.getElementById('docTypeTag').innerText = docTypeName;

      // Meta & Badges
      document.getElementById('resModelBadge').innerText = result.model_used;
      const modeBadge = document.getElementById('resModeBadge');
      if (result.is_mock) {
        modeBadge.className = 'badge';
        modeBadge.style.color = '#fbbf24';
        modeBadge.innerText = 'Snapshot Mock';
      } else {
        modeBadge.className = 'badge green';
        modeBadge.innerText = 'Authentic Local GPU';
      }
      document.getElementById('resLatencyBadge').innerText = `⏱️ ${(result.latency_ms / 1000).toFixed(2)}s`;

      // Pipeline Latency Breakdown (Component 5)
      const timingBanner = document.getElementById('pipelineTimingsBanner');
      const t = result.timings || result.timing;
      if (t) {
        timingBanner.style.display = 'flex';
        document.getElementById('timingOcrTag').innerText = `OCR: ${t.ocr_ms.toFixed(1)} ms`;
        document.getElementById('timingLlmTag').innerText = `LLM: ${t.llm_ms.toFixed(1)} ms`;
        document.getElementById('timingValTag').innerText = `Val: ${t.validation_ms.toFixed(1)} ms`;
        const totalMs = t.total_ms !== undefined ? t.total_ms : (t.total_pipeline_ms || 0);
        document.getElementById('timingTotalTag').innerText = `Total: ${totalMs.toFixed(1)} ms`;
      } else if (result.latency_ms !== undefined) {
        timingBanner.style.display = 'flex';
        document.getElementById('timingOcrTag').innerText = 'OCR: -';
        document.getElementById('timingLlmTag').innerText = `LLM: ${result.latency_ms.toFixed(1)} ms`;
        const valMs = result.validation?.validation_latency_ms || 0;
        document.getElementById('timingValTag').innerText = `Val: ${valMs.toFixed(1)} ms`;
        document.getElementById('timingTotalTag').innerText = `Total: ${(result.latency_ms + valMs).toFixed(1)} ms`;
      } else {
        timingBanner.style.display = 'none';
      }

      // Chain of Thought
      const cotAccordion = document.getElementById('cotAccordion');
      if (result.thinking_process) {
        cotAccordion.style.display = 'block';
        document.getElementById('cotBody').innerText = result.thinking_process;
      } else {
        cotAccordion.style.display = 'none';
      }

      // Component 4: Validation Card & Quality Gates
      const val = result.validation;
      const valCard = document.getElementById('validationResultCard');
      const valStatusBadge = document.getElementById('resValStatusBadge');

      if (val) {
        valCard.style.display = 'block';
        valStatusBadge.style.display = 'inline-flex';

        // Overall status badge
        const badgeEl = document.getElementById('valOverallBadge');
        if (val.status === 'PASSED') {
          badgeEl.className = 'badge green';
          badgeEl.innerText = '✓ ผ่านการตรวจสอบทั้งหมด (PASSED)';
          valStatusBadge.className = 'badge green';
          valStatusBadge.innerText = '✓ ผ่านเกณฑ์';
        } else if (val.status === 'WARNING') {
          badgeEl.className = 'badge';
          badgeEl.style.color = '#fbbf24';
          badgeEl.style.borderColor = 'rgba(245, 158, 11, 0.4)';
          badgeEl.innerText = '⚠️ ผ่านแบบมีข้อสังเกต (WARNING)';
          valStatusBadge.className = 'badge';
          valStatusBadge.style.color = '#fbbf24';
          valStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
          valStatusBadge.innerText = '⚠️ มีข้อสังเกต';
        } else {
          badgeEl.className = 'badge';
          badgeEl.style.color = '#f87171';
          badgeEl.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          badgeEl.innerText = '❌ ตรวจพบข้อผิดพลาด (ERROR)';
          valStatusBadge.className = 'badge';
          valStatusBadge.style.color = '#f87171';
          valStatusBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          valStatusBadge.innerText = '❌ มีข้อผิดพลาด';
        }

        // 1. Date Gate
        const dateDesc = document.getElementById('gateDateDesc');
        if (val.date_report && val.date_report.is_valid) {
          dateDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">✓ ${val.date_report.iso_date}</span> <small style="color: var(--text-muted);">(${val.date_report.thai_formatted})</small>`;
        } else if (val.date_report && val.date_report.raw_date) {
          dateDesc.innerHTML = `<span style="color: #fbbf24; font-weight: 600;">⚠️ ${val.date_report.raw_date}</span>`;
        } else {
          dateDesc.innerHTML = `<span style="color: var(--text-muted);">- ไม่ระบุวันที่ -</span>`;
        }

        // 2. Tax ID Gate
        const taxDesc = document.getElementById('gateTaxDesc');
        if (val.tax_id_report && val.tax_id_report.is_valid) {
          taxDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">✓ ${val.tax_id_report.formatted_id}</span> <small style="color: #94a3b8;">(Mod 11 Checksum ✓)</small>`;
        } else if (val.tax_id_report && val.tax_id_report.raw_id) {
          taxDesc.innerHTML = `<span style="color: #f87171; font-weight: 600;">❌ ${val.tax_id_report.raw_id}</span> <small style="color: #fca5a5;">(${val.tax_id_report.error_message || 'ไม่ผ่าน'})</small>`;
        } else {
          taxDesc.innerHTML = `<span style="color: var(--text-muted);">- ไม่มีระบุในเอกสาร -</span>`;
        }

        // 3. Math Gate
        const mathDesc = document.getElementById('gateMathDesc');
        if (val.math_report && val.math_report.is_balanced) {
          mathDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">✓ ยอดเงินสอดคล้อง</span> <small style="color: var(--text-muted); display: block;">${val.math_report.details || ''}</small>`;
        } else if (val.math_report) {
          mathDesc.innerHTML = `<span style="color: #f87171; font-weight: 600;">❌ ยอดเงินคลาดเคลื่อน</span> <small style="color: #fca5a5; display: block;">${val.math_report.details || ''}</small>`;
        } else {
          mathDesc.innerHTML = `<span style="color: var(--text-muted);">- ไม่มียอดเงินที่ต้องคำนวณ -</span>`;
        }

        // 4. University Policy Gate
        const policyDesc = document.getElementById('gatePolicyDesc');
        const policyIssues = (val.issues || []).filter(i => i.code === 'MISSING_REQUIRED_FIELD' || i.code === 'PETTY_CASH_EXCEEDS_LIMIT' || i.code === 'EMPTY_EXPENSE_ITEMS');
        if (policyIssues.length === 0) {
          policyDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">✓ ครบถ้วนตามระเบียบมหาวิทยาลัย</span>`;
        } else {
          const hasErr = policyIssues.some(i => i.severity === 'ERROR');
          policyDesc.innerHTML = `<span style="color: ${hasErr ? '#f87171' : '#fbbf24'}; font-weight: 600;">${hasErr ? '❌ ขาดข้อมูลจำเป็น' : '⚠️ มีเงื่อนไขเตือน'} (${policyIssues.length} จุด)</span>`;
        }

        // Issues List
        const issuesContainer = document.getElementById('valIssuesContainer');
        const issuesList = document.getElementById('valIssuesList');
        issuesList.innerHTML = '';
        if (val.issues && val.issues.length > 0) {
          issuesContainer.style.display = 'flex';
          val.issues.forEach(issue => {
            const item = document.createElement('div');
            item.className = issue.severity === 'ERROR' ? 'val-issue-item val-issue-error' : 'val-issue-item val-issue-warning';
            item.innerHTML = `
              <span style="font-size: 1rem;">${issue.severity === 'ERROR' ? '❌' : '⚠️'}</span>
              <div>
                <b>[${issue.field}]</b> ${issue.message}
              </div>
            `;
            issuesList.appendChild(item);
          });
        } else {
          issuesContainer.style.display = 'none';
        }
      } else {
        valCard.style.display = 'none';
        valStatusBadge.style.display = 'none';
      }

      // Dynamic Grid of Extracted Fields
      const fieldGrid = document.getElementById('fieldGrid');
      fieldGrid.innerHTML = '';

      const fieldLabels = {
        doc_no: "1. เลขที่เอกสาร",
        doc_date: "2. วันที่ทำเอกสาร",
        title: "3. เรื่อง",
        requester: "4. ผู้ทำการเบิก / หน่วยงาน",
        ref_doc_no: "ตามหนังสือเลขที่ (อ้างอิง)",
        ref_memo_no: "แนบท้ายบันทึกเลขที่",
        disbursement_type: "ประเภทการเบิกจ่าย",
        transfer_destination: "โอนเงินไปที่ใด",
        submission_date: "วันที่ส่งเอกสาร",
        claim_date: "วันที่ขอรับเงิน",
        payer: "ผู้จ่ายเงิน",
        inspectors: "ผู้ตรวจรับพัสดุ / คณะกรรมการ",
        inspection_date: "วันที่ตรวจรับ",
        procurement_reason: "เหตุผลความจำเป็นที่ต้องจัดหา",
        total_budget: "วงเงินงบประมาณที่ใช้ (บาท)",
        total_approved_amount: "ยอดรวมที่อนุมัติ (บาท)",
        required_date: "เวลาที่ต้องใช้พัสดุ",
        vendor_name: "ชื่อร้านค้า / ผู้ขาย",
        vendor_tax_id: "เลขประจำตัวผู้เสียภาษี 13 หลัก",
        vendor_branch: "สาขา",
        vendor_address: "ที่อยู่",
        customer_name: "ชื่อผู้ซื้อ / ลูกค้า",
        customer_tax_id: "เลขประจำตัวผู้ซื้อ",
        invoice_no: "เลขที่ใบเสร็จ / เอกสาร",
        subtotal: "รวมเป็นเงิน / ค่าสินค้า (บาท)",
        vat: "ภาษีมูลค่าเพิ่ม (บาท)",
        total_amount: "ยอดรวมทั้งสิ้น (บาท)"
      };

      const fieldsObj = result.fields || {};
      let fieldCount = 0;
      for (const [key, val] of Object.entries(fieldsObj)) {
        if (Array.isArray(val) || key === 'line_items' || key === 'expense_items' || key === 'approval_items' || key === 'item_details') {
          continue;
        }
        if (val === null || val === undefined) continue;

        fieldCount++;
        const itemDiv = document.createElement('div');
        itemDiv.className = 'field-item';
        const labelText = fieldLabels[key] || key;
        const displayVal = (typeof val === 'number') ? formatCurrency(val) : String(val);

        itemDiv.innerHTML = `
          <span class="field-label">${labelText}</span>
          <span class="field-value">${displayVal}</span>
        `;
        fieldGrid.appendChild(itemDiv);
      }

      if (fieldCount === 0) {
        fieldGrid.innerHTML = '<div style="color: var(--text-muted); font-size: 0.8rem; padding: 0.5rem;">ไม่มีฟิลด์ข้อมูลเดี่ยว</div>';
      }

      // Line Items / Expense Items
      const items = result.expense_items || result.line_items || [];
      const lineItemsCard = document.getElementById('lineItemsCard');
      if (items.length > 0) {
        lineItemsCard.style.display = 'block';
        const tbody = document.getElementById('lineItemsTableBody');
        tbody.innerHTML = '';
        document.getElementById('valItemCount').innerText = `${items.length} รายการ`;

        items.forEach((item, idx) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td style="color: var(--text-muted);">${item.item_no || (idx + 1)}</td>
            <td style="font-weight: 500;">${item.description}</td>
            <td class="num-col">${item.quantity !== null && item.quantity !== undefined ? item.quantity : '-'}</td>
            <td style="color: var(--text-muted);">${item.unit || '-'}</td>
            <td class="num-col">${item.unit_price !== null && item.unit_price !== undefined ? formatCurrency(item.unit_price) : '-'}</td>
            <td class="num-col" style="font-weight: 600; color: #60a5fa;">${item.total_price !== null && item.total_price !== undefined ? formatCurrency(item.total_price) : '-'}</td>
          `;
          tbody.appendChild(tr);
        });
      } else {
        lineItemsCard.style.display = 'none';
      }

      // Totals Box
      const total = result.total_amount;
      if (total !== null && total !== undefined) {
        document.getElementById('valTotal').innerText = `฿${formatCurrency(total)}`;
        document.getElementById('totalsBoxContainer').style.display = 'flex';
      } else {
        document.getElementById('totalsBoxContainer').style.display = 'none';
      }

      if (result.subtotal !== null && result.subtotal !== undefined) {
        document.getElementById('subtotalRow').style.display = 'block';
        document.getElementById('valSubtotal').innerText = `฿${formatCurrency(result.subtotal)}`;
      } else {
        document.getElementById('subtotalRow').style.display = 'none';
      }

      if (result.vat !== null && result.vat !== undefined) {
        document.getElementById('vatRow').style.display = 'block';
        document.getElementById('valVat').innerText = `฿${formatCurrency(result.vat)}`;
      } else {
        document.getElementById('vatRow').style.display = 'none';
      }

      // Mathematical Verification Check
      const mathBadge = document.getElementById('mathValidationBadge');
      if (result.subtotal && result.vat && total) {
        const expectedTotal = Math.round((result.subtotal + result.vat) * 100) / 100;
        const actualTotal = Math.round(total * 100) / 100;
        if (Math.abs(expectedTotal - actualTotal) < 0.05) {
          mathBadge.style.display = 'inline-flex';
          mathBadge.className = 'badge green';
          mathBadge.innerText = `✓ ตรวจสอบยอดเงินถูกต้อง (Subtotal + VAT = Total: ฿${formatCurrency(actualTotal)})`;
        } else {
          mathBadge.style.display = 'inline-flex';
          mathBadge.className = 'badge';
          mathBadge.style.color = '#f87171';
          mathBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          mathBadge.innerText = `⚠️ ยอดระบุ: ฿${formatCurrency(actualTotal)} (ต่างจาก Subtotal + VAT: ฿${formatCurrency(expectedTotal)})`;
        }
      } else if (total) {
        mathBadge.style.display = 'inline-flex';
        mathBadge.className = 'badge green';
        mathBadge.innerText = `✓ ยอดสุทธิระบุในเอกสาร: ฿${formatCurrency(total)}`;
      } else {
        mathBadge.style.display = 'none';
      }

      // Extraction Tab Badge
      const extBadge = document.getElementById('extractionBadge');
      extBadge.style.display = 'inline-flex';
      extBadge.innerText = 'สกัดแล้ว';
    }

    function formatCurrency(val) {
      if (val === null || val === undefined) return '0.00';
      return Number(val).toLocaleString('th-TH', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function toggleCot() {
      const body = document.getElementById('cotBody');
      const icon = document.getElementById('cotToggleIcon');
      if (body.style.display === 'none') {
        body.style.display = 'block';
        icon.innerText = '▼';
      } else {
        body.style.display = 'none';
        icon.innerText = '▶';
      }
    }

    function copyExtractionJson() {
      if (!lastExtractionResult) return;
      navigator.clipboard.writeText(JSON.stringify(lastExtractionResult, null, 2))
        .then(() => alert('คัดลอก JSON ผลลัพธ์ลง Clipboard เรียบร้อยแล้ว!'))
        .catch(err => alert('ไม่สามารถคัดลอกได้: ' + err));
    }

    // Initialize LLM status on load
    window.addEventListener('DOMContentLoaded', loadLLMStatus);

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

    /* Component 5: Model Benchmarking Modal Logic */
    let lastBenchmarkReport = null;

    function openBenchmarkModal() {
      document.getElementById('benchmarkModal').classList.add('active');
    }

    function closeBenchmarkModal() {
      document.getElementById('benchmarkModal').classList.remove('active');
    }

    async function executeBenchmark() {
      const btn = document.getElementById('btnRunBenchmark');
      const loader = document.getElementById('bmLoading');
      const resultsArea = document.getElementById('bmResultsArea');
      const docType = document.getElementById('bmDocTypeSelect').value || 'general_receipt';
      const tempVal = parseFloat(document.getElementById('bmTempSelect').value || '0.0');

      btn.disabled = true;
      loader.style.display = 'flex';
      resultsArea.style.display = 'none';

      try {
        const res = await fetch('/api/v1/benchmark', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            document_type: docType,
            temperature: tempVal,
            force_mock: false
          })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Benchmark failed');
        }

        const report = await res.json();
        lastBenchmarkReport = report;
        document.getElementById('bmMarkdownView').innerText = report.markdown_table;
        resultsArea.style.display = 'flex';
      } catch (err) {
        alert('เกิดข้อผิดพลาดในการรัน Benchmark: ' + err.message);
      } finally {
        btn.disabled = false;
        loader.style.display = 'none';
      }
    }

    function copyBenchmarkMarkdown() {
      if (!lastBenchmarkReport || !lastBenchmarkReport.markdown_table) return;
      navigator.clipboard.writeText(lastBenchmarkReport.markdown_table)
        .then(() => alert('คัดลอกตาราง Markdown ลง Clipboard เรียบร้อยแล้ว! นำไปวางในสไลด์หรือรายงานได้ทันที'))
        .catch(err => alert('ไม่สามารถคัดลอกได้: ' + err));
    }
  </script>

  <!-- Model Benchmark Modal Dialog (Component 5) -->
  <div class="modal-overlay" id="benchmarkModal">
    <div class="modal-card">
      <div class="modal-header">
        <h3>📊 เปรียบเทียบโมเดล AI สกัดข้อมูลทางการเงิน (LLM Benchmarking)</h3>
        <button class="btn-icon" onclick="closeBenchmarkModal()" style="font-size: 1.1rem;">✕</button>
      </div>
      <div class="modal-body">
        <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.5rem;">
          ทดสอบวัดประสิทธิภาพเชิงปริมาณ (Latency, Field Completeness, Validation Pass Rate, Math Reconciliation) ระหว่าง Local LLM แต่ละรุ่นตามเกณฑ์รายงานผลงานวิจัย / สัมมนาปริญญานิพนธ์
        </p>

        <div style="display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; background: rgba(0,0,0,0.25); padding: 0.75rem; border-radius: 8px; border: 1px solid var(--border);">
          <div>
            <label style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 0.25rem;">ประเภทเอกสารทดสอบ:</label>
            <select id="bmDocTypeSelect" class="search-input" style="width: auto; min-width: 220px; padding: 0.25rem 0.5rem;"></select>
          </div>
          <div>
            <label style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 0.25rem;">อุณหภูมิ (Temperature):</label>
            <select id="bmTempSelect" class="search-input" style="width: auto; padding: 0.25rem 0.5rem;">
              <option value="0.0" selected>0.0 (Greedy / Deterministic)</option>
              <option value="0.1">0.1</option>
            </select>
          </div>
          <div style="margin-top: 1rem;">
            <button class="btn" id="btnRunBenchmark" onclick="executeBenchmark()">
              🚀 เริ่มการทดสอบ (Run Benchmark)
            </button>
          </div>
        </div>

        <div id="bmLoading" style="display: none; align-items: center; justify-content: center; gap: 0.75rem; padding: 2rem; color: #94a3b8;">
          <div class="spinner" style="display: block; width: 24px; height: 24px;"></div>
          <span>กำลังรันการทดสอบและวิเคราะห์ผลเชิงสถิติ... (โปรดรอสักครู่)</span>
        </div>

        <div id="bmResultsArea" style="display: none; flex-direction: column; gap: 0.8rem;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #e2e8f0;">📋 ตารางเปรียบเทียบผลลัพธ์ (Markdown Format)</span>
            <button class="btn btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="copyBenchmarkMarkdown()">
              📋 คัดลอก Markdown สำหรับสไลด์ / เล่มรายงาน
            </button>
          </div>
          <div class="code-scroll-container" style="max-height: 260px; background: rgba(0,0,0,0.4); border-radius: 8px; border: 1px solid var(--border);">
            <pre id="bmMarkdownView" style="margin: 0;"></pre>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" onclick="closeBenchmarkModal()">ปิดหน้าต่าง</button>
      </div>
    </div>
  </div>
</body>
</html>
"""


from starlette.concurrency import run_in_threadpool

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

    perception = await run_in_threadpool(ocr_engine.process_image, processed_image, page_number=1)

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
async def get_sample():
    sample_path = ROOT_DIR / "tests" / "output" / "sample_receipt.png"
    if not sample_path.exists():
        from tests.test_components_1_2 import create_sample_receipt_image
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        img = create_sample_receipt_image()
        img.save(sample_path)
    else:
        img = Image.open(sample_path)

    perception = await run_in_threadpool(ocr_engine.process_image, img, page_number=1)

    return {
        "filename": "sample_receipt.png",
        "width": img.width,
        "height": img.height,
        "image_base64": pil_to_base64(img),
        "text_blocks": [block.model_dump() for block in perception.text_blocks],
        "raw_text": perception.raw_text,
        "llm_markdown": perception.to_llm_markdown()
    }


@app.get("/api/llm/status")
async def get_llm_status():
    """Check Ollama connectivity and installed models."""
    return await run_in_threadpool(llm_extractor.get_status)


class LLMExtractRequest(BaseModel):
    ocr_markdown: str
    document_type: str = "general_receipt"
    model_name: Optional[str] = None
    temperature: float = 0.0
    force_mock: bool = False


@app.post("/api/llm/extract")
async def extract_financial_data(req: LLMExtractRequest):
    """Extract structured financial data with Local LLM (or mock fallback)."""
    try:
        result = await run_in_threadpool(
            llm_extractor.extract,
            ocr_markdown=req.ocr_markdown,
            document_type=req.document_type,
            model_name=req.model_name,
            temperature=req.temperature,
            force_mock=req.force_mock
        )
        return result.model_dump()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/validate")
async def validate_document_data(data: dict):
    """Component 4: Standalone document validation, normalization & business rules endpoint."""
    try:
        from src.schemas import TypeDirectedExtractionResult
        extracted = TypeDirectedExtractionResult.model_validate(data)
        val_result = validator.validate(extracted)
        return val_result.model_dump()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# =============================================================================
# Component 5: Production REST API v1 Endpoints (OpenAPI / Swagger Documented)
# =============================================================================

@app.post(
    "/api/v1/extract",
    response_model=FullPipelineResult,
    tags=["End-to-End Extraction Pipeline"],
    summary="Full End-to-End Financial Document Extraction Pipeline",
    description="Accepts document file (PDF or Image) and document type. Executes Ingestion -> OCR -> LLM -> Validation & Normalization in a single call."
)
async def api_v1_extract(
    file: UploadFile = File(..., description="Document file (PDF or image)"),
    document_type: str = Form(DocumentType.GENERAL_RECEIPT.value, description="Target document type (1 of 10 types)"),
    model_name: Optional[str] = Form(None, description="Ollama model name (default: qwen2.5:3b)"),
    auto_deskew: bool = Form(False, description="Enable automatic image deskewing"),
    force_mock: bool = Form(False, description="Force snapshot mock for offline evaluation"),
    temperature: float = Form(0.0, description="LLM temperature (0.0 for zero hallucination)")
):
    """Component 5: Complete End-to-End Extraction Pipeline."""
    try:
        file_bytes = await file.read()
        result = await run_in_threadpool(
            pipeline.process,
            file_bytes=file_bytes,
            filename=file.filename or "document.png",
            document_type=document_type,
            model_name=model_name,
            auto_deskew=auto_deskew,
            force_mock=force_mock,
            temperature=temperature
        )
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post(
    "/api/v1/validate",
    response_model=DocumentValidationResult,
    tags=["Data Validation & Rules"],
    summary="Standalone Document Validation & Normalization",
    description="Validates extracted document fields against Thai date normalization, Thai Tax ID Mod 11, financial math reconciliation, and university business rules."
)
async def api_v1_validate(data: TypeDirectedExtractionResult):
    """Component 5: Standalone data validation and normalization endpoint."""
    try:
        val_result = await run_in_threadpool(validator.validate, data)
        return val_result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post(
    "/api/v1/benchmark",
    response_model=BenchmarkReport,
    tags=["Model Benchmarking"],
    summary="Run Comparative Model Benchmark",
    description="Executes comparative benchmarking on financial extraction across specified LLM models. Returns latency, accuracy, and Markdown comparison table."
)
async def api_v1_benchmark(req: BenchmarkRequest):
    """Component 5: Comparative model benchmarking engine."""
    try:
        report = await run_in_threadpool(
            benchmarker.run_benchmark,
            models=req.models,
            document_type=req.document_type,
            temperature=req.temperature,
            include_mock_baseline=not req.force_mock
        )
        return report
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    tags=["System Health"],
    summary="Information Extraction Pipeline Health Check",
    description="Returns readiness status and metadata for all 4 pipeline components."
)
async def api_v1_health():
    """Component 5: System health check and component monitoring."""
    return await run_in_threadpool(pipeline.get_health_status)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Thai Financial Document OCR UI...")
    print("👉 Open your browser at: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

