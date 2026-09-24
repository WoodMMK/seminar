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

from src.pp_chatocr_engine import PPChatOCREngine
chat_ocr_engine = PPChatOCREngine()

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

    /* Flow Selector Segmented Bar */
    .flow-selector-bar {
      display: flex;
      flex-shrink: 0;
      gap: 0.5rem;
    }
    .flow-segmented-control {
      display: flex;
      width: 100%;
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 3px;
      gap: 6px;
    }
    .flow-btn {
      flex: 1;
      display: flex;
      align-items: center;
      gap: 0.65rem;
      padding: 0.5rem 0.9rem;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 7px;
      cursor: pointer;
      color: var(--text-muted);
      text-align: left;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .flow-btn:hover {
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
    }
    .flow-btn.active#flowBtnCustom {
      background: rgba(37, 99, 235, 0.16);
      border-color: rgba(59, 130, 246, 0.45);
      color: #93c5fd;
      box-shadow: 0 2px 10px rgba(37, 99, 235, 0.25);
    }
    .flow-btn.active#flowBtnChatOCR {
      background: rgba(168, 85, 247, 0.18);
      border-color: rgba(168, 85, 247, 0.5);
      color: #e9d5ff;
      box-shadow: 0 2px 10px rgba(168, 85, 247, 0.25);
    }
    .flow-icon {
      font-size: 1.25rem;
      line-height: 1;
      flex-shrink: 0;
    }
    .flow-text {
      display: flex;
      flex-direction: column;
      gap: 1px;
    }
    .flow-title {
      font-size: 0.82rem;
      font-weight: 600;
    }
    .flow-desc {
      font-size: 0.69rem;
      opacity: 0.75;
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

    /* PP-ChatOCRv4 Interactive Chat & Knowledge Payload (Component 6) */
    .chatocr-container {
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 0.8rem;
      gap: 0.8rem;
    }
    .chatocr-ctrl-box {
      background: rgba(0, 0, 0, 0.28);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 0.9rem;
      flex-shrink: 0;
    }
    .preset-chips-group {
      display: flex;
      gap: 0.4rem;
      flex-wrap: wrap;
    }
    .preset-chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 9999px;
      padding: 0.25rem 0.6rem;
      font-size: 0.73rem;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
    }
    .preset-chip:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: rgba(59, 130, 246, 0.5);
      color: #93c5fd;
      transform: translateY(-1px);
    }
    .chatocr-answers-card {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 0.9rem;
    }
    .chatocr-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.6rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      padding-bottom: 0.45rem;
    }
    .chatocr-qa-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }
    .chatocr-qa-item {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 6px;
      padding: 0.55rem 0.8rem;
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      transition: border-color 0.15s;
    }
    .chatocr-qa-item:hover {
      border-color: rgba(59, 130, 246, 0.35);
    }
    .chatocr-qa-item.crosscheck-item {
      background: rgba(56, 189, 248, 0.05);
      border: 1px solid rgba(56, 189, 248, 0.28);
      border-left: 4px solid #38bdf8;
    }
    .chatocr-qa-item.crosscheck-item:hover {
      border-color: rgba(56, 189, 248, 0.55);
    }
    .chatocr-q-title {
      font-size: 0.76rem;
      font-weight: 600;
      color: #93c5fd;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .chatocr-a-body {
      font-size: 0.82rem;
      color: #f8fafc;
      font-weight: 500;
      line-height: 1.4;
    }
    .knowledge-payload-container {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 8px;
      padding: 0.85rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .payload-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    @media (max-width: 900px) {
      .payload-grid { grid-template-columns: 1fr; }
    }
    .payload-card {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 6px;
      padding: 0.65rem;
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }
    .payload-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.5rem;
    }
    .chatocr-qa-input {
      width: 100%;
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 5px;
      color: #f1f5f9;
      font-size: 0.8rem;
      padding: 0.4rem 0.6rem;
      font-family: inherit;
      resize: vertical;
      line-height: 1.4;
      box-sizing: border-box;
      transition: all 0.2s ease;
    }
    .chatocr-qa-input:focus {
      outline: none;
      border-color: #38bdf8;
      background: rgba(0, 0, 0, 0.55);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25);
    }
    .field-editor-box {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(59, 130, 246, 0.35);
      border-radius: 8px;
      padding: 0.75rem 0.9rem;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    /* Dashboard & Case Traceability Styles */
    .dashboard-view {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      padding: 0.8rem;
      flex: 1;
      min-height: 0;
      overflow-y: auto;
    }
    .dashboard-toolbar {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.6rem 0.85rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.6rem;
      flex-wrap: wrap;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.6rem;
    }
    @media (max-width: 1024px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi-card {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .kpi-title {
      font-size: 0.68rem;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
      letter-spacing: 0.03em;
    }
    .kpi-value {
      font-size: 1.25rem;
      font-weight: 700;
      color: #f8fafc;
      line-height: 1.2;
    }
    .kpi-sub {
      font-size: 0.69rem;
      color: #94a3b8;
    }
    .case-card {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
      transition: all 0.2s ease;
    }
    .case-card:hover {
      border-color: rgba(59, 130, 246, 0.45);
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .case-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .case-pno-badge {
      font-size: 0.76rem;
      font-weight: 700;
      padding: 0.18rem 0.55rem;
      border-radius: 6px;
      background: rgba(59, 130, 246, 0.16);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.4);
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
    }
    .progress-bar-wrap {
      background: rgba(0, 0, 0, 0.4);
      border-radius: 999px;
      height: 8px;
      overflow: hidden;
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .progress-bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.3s ease;
    }
    .checklist-row {
      display: flex;
      gap: 0.45rem;
      flex-wrap: wrap;
      align-items: center;
    }
    .checklist-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.2rem 0.5rem;
      border-radius: 5px;
      font-size: 0.69rem;
      font-weight: 500;
    }
    .checklist-pill.ok {
      background: rgba(16, 185, 129, 0.14);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .checklist-pill.missing {
      background: rgba(255, 255, 255, 0.03);
      color: #64748b;
      border: 1px dashed rgba(255, 255, 255, 0.12);
    }
    .case-doc-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.77rem;
      margin-top: 0.3rem;
    }
    .case-doc-table th {
      background: rgba(0, 0, 0, 0.35);
      padding: 0.38rem 0.55rem;
      color: var(--text-muted);
      text-align: left;
      font-weight: 600;
    }
    .case-doc-table td {
      padding: 0.38rem 0.55rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
    }
    .case-doc-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
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
      <a href="/docs" target="_blank" class="badge" style="color: #60a5fa; border-color: rgba(96, 165, 250, 0.4); text-decoration: none; font-weight: 500;">Swagger API Docs (/docs)</a>
      <span class="badge green">● Model: th_PP-OCRv5_mobile_rec</span>
      <span id="headerDbBadge" class="badge" style="color: #93c5fd; border-color: rgba(59, 130, 246, 0.4);">PostgreSQL: เชื่อมต่อ...</span>
      <span class="badge">Privacy 100% On-Premises</span>
    </div>
  </header>

  <main>
    <!-- Flow Architecture Segmented Switcher -->
    <div class="flow-selector-bar">
      <div class="flow-segmented-control">
        <button class="flow-btn active" id="flowBtnCustom" onclick="setAppFlow('custom')">
          <div class="flow-text">
            <span class="flow-title">โฟลว์ที่ 1: Custom Pipeline (Hybrid RT-DETR + PaddleOCR + LLM + Validator)</span>
            <span class="flow-desc">สกัดโครงสร้างข้อความ, ทำ Context Markdown, ตรวจสอบกฎเบิกจ่าย และความถูกต้องทางบัญชี</span>
          </div>
        </button>
        <button class="flow-btn" id="flowBtnChatOCR" onclick="setAppFlow('chatocr')">
          <div class="flow-text">
            <span class="flow-title">โฟลว์ที่ 2: PP-ChatOCRv4 (PaddleX สำเร็จรูป + Type-Directed Prompts)</span>
            <span class="flow-desc">สกัด 9 ประเภทเบิกจ่ายราชการ + 1 ใบเสร็จทั่วไปอัตโนมัติ สร้าง Knowledge Payload สำหรับ Vector DB</span>
          </div>
        </button>
      </div>
    </div>

    <!-- Top Compact Controls Toolbar -->
    <div class="controls-bar">
      <div class="file-input-group">
        <input type="file" id="fileInput" accept="image/*,.pdf" style="display: none;">
        <button class="btn" id="btnUploadFile" onclick="document.getElementById('fileInput').click()">
          เลือกไฟล์ภาพหรือ PDF
        </button>
        <button class="btn btn-secondary" id="btnLoadSample" onclick="loadSampleReceipt()">
          ทดสอบบิลตัวอย่าง (Sample)
        </button>
        <label class="checkbox-label" id="deskewCheckLabel">
          <input type="checkbox" id="deskewCheck"> ปรับมุมเอียงอัตโนมัติ (Deskew)
        </label>
      </div>
      <div style="display: flex; align-items: center; gap: 0.75rem;">
        <span id="activeFlowBadge" class="badge" style="background: rgba(37, 99, 235, 0.15); color: #93c5fd; border: 1px solid rgba(37, 99, 235, 0.3); font-size: 0.75rem;">
          โฟลว์: Custom Pipeline
        </span>
        <div class="spinner" id="loadingSpinner"></div>
        <span id="statusText" style="font-size: 0.82rem; color: var(--text-muted);">พร้อมใช้งาน</span>
      </div>
    </div>

    <!-- Dual Workspace: Pinned Left Image & Dedicated Scroll Right Debug -->
    <div class="workspace">
      <!-- Left: Visual Document Stage with Bounding Box Overlay -->
      <div class="panel">
        <div class="panel-header">
          <h2>การแสดงผลเอกสาร & Bounding Boxes</h2>
          <div class="tool-group">
            <label class="checkbox-label" style="margin-right: 0.4rem;">
              <input type="checkbox" id="toggleBboxCheck" checked onchange="toggleBoundingBoxes(this.checked)"> แสดงกรอบ
            </label>
            <button class="btn-icon" onclick="zoomChange(-0.15)" title="Zoom Out">-</button>
            <button class="btn-icon" id="zoomLabel" onclick="zoomReset()" title="Reset Zoom">100%</button>
            <button class="btn-icon" onclick="zoomChange(0.15)" title="Zoom In">+</button>
            <button class="btn-icon" onclick="zoomFit()" title="Fit Width">Fit</button>
            <button class="btn-icon" onclick="rotateCurrentDoc(90)" title="หมุนเอกสาร 90 องศาตามเข็ม">หมุน 90°</button>
            <span id="pageInfo" class="badge">0x0 px</span>
          </div>
        </div>
        <div class="viewer-stage" id="viewerStage">
          <div class="canvas-container" id="canvasContainer">
            <img id="docImage" src="" alt="Document Preview" style="display: none;">
            <div class="bbox-overlay" id="bboxOverlay"></div>
          </div>
          <div id="emptyState" style="color: var(--text-muted); font-size: 0.88rem; align-self: center;">
            กรุณาเลือกไฟล์ภาพบิล หรือกดปุ่ม "ทดสอบบิลตัวอย่าง" เพื่อเริ่มต้น
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
            <span>สกัดข้อมูลการเงิน (AI)</span>
            <span class="badge green" id="extractionBadge" style="display: none;">Ready</span>
          </button>
          <button class="tab-btn" onclick="switchTab('chatocr')" id="tabBtnChatOCR">
            <span>PP-ChatOCRv4 (Chat)</span>
            <span class="badge green" id="chatocrBadge" style="display: none;">Ready</span>
          </button>
          <button class="tab-btn" onclick="switchTab('json')" id="tabBtnJson">Raw API JSON</button>
          <button class="tab-btn" onclick="switchTab('dashboard')" id="tabBtnDashboard">
            <span>แดชบอร์ดติดตามเรื่อง (Dossier)</span>
            <span class="badge" id="dashboardCountBadge" style="background: rgba(59, 130, 246, 0.2); color: #93c5fd;">0 เรื่อง</span>
          </button>
        </div>

        <!-- Tab 1: Scrollable Text Blocks List -->
        <div class="tab-content active" id="tabBlocks">
          <div class="filter-bar">
            <input type="text" id="filterInput" class="search-input" placeholder="ค้นหาข้อความใน Blocks..." oninput="filterBlocks(this.value)">
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
                <label style="font-size: 0.78rem; color: var(--text-muted); font-weight: 600;">ประเภทเอกสาร (Manual Input):</label>
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
                  เปรียบเทียบโมเดล (Benchmark)
                </button>
                <button class="btn" id="btnExtractLLM" onclick="runLLMExtraction()">
                  สกัดข้อมูลด้วย AI
                </button>
              </div>
            </div>

            <div id="extractionEmpty" style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 2.5rem;">
              กรุณาเลือกประเภทเอกสารใน Dropdown แล้วกดปุ่ม <b>"สกัดข้อมูลด้วย AI"</b> เพื่อให้โมเดลสกัดข้อมูลเฉพาะประเภท
            </div>

            <div id="extractionResultsArea" style="display: none; flex-direction: column; gap: 0.75rem;">
              <!-- Meta & Latency Banner -->
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                  <span id="resDocTypeBadge" class="badge green">เอกสารขออนุมัติหลักการ</span>
                  <span id="resModelBadge" class="badge">qwen2.5:3b</span>
                  <span id="resModeBadge" class="badge">Local GPU</span>
                  <span id="resLatencyBadge" class="badge">- ms</span>
                  <span id="resValStatusBadge" class="badge green" style="display: none;">[ผ่าน] ตรวจสอบผ่าน</span>
                </div>
                <div style="display: flex; gap: 0.4rem;">
                  <button class="btn btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="copyExtractionJson()">คัดลอก JSON</button>
                </div>
              </div>

              <!-- Pipeline Timing Stages Breakdown (Component 5) -->
              <div class="timing-bar" id="pipelineTimingsBanner" style="display: none;">
                <span style="font-weight: 600; color: #94a3b8;">Pipeline Latency:</span>
                <span id="timingOcrTag" class="badge">OCR: - ms</span>
                <span id="timingLlmTag" class="badge">LLM: - ms</span>
                <span id="timingValTag" class="badge">Validation: - ms</span>
                <span id="timingTotalTag" class="badge green" style="font-weight: 600;">Total: - ms</span>
              </div>

              <!-- Collapsible CoT Reasoning -->
              <div class="cot-accordion" id="cotAccordion" style="display: none;">
                <div class="cot-header" onclick="toggleCot()">
                  <span>ลำดับความคิดวิเคราะห์ (Chain-of-Thought / &lt;think&gt;)</span>
                  <span id="cotToggleIcon">▼</span>
                </div>
                <div class="cot-body" id="cotBody"></div>
              </div>

              <!-- Component 4: Validation & Rules Engine Status Card -->
              <div class="field-card" id="validationResultCard" style="display: none;">
                <div class="field-card-title">
                  <span>การตรวจสอบความถูกต้อง & กฎระเบียบ (Component 4 Rules Engine)</span>
                  <span id="valOverallBadge" class="badge green">PASSED</span>
                </div>
                
                <!-- 4 Quality Gates Grid -->
                <div class="val-gate-grid">
                  <div class="val-gate-item" id="gateDateBox">
                    <span class="gate-title">รูปแบบวันที่ (Thai Date Normalization)</span>
                    <span class="gate-desc" id="gateDateDesc">-</span>
                  </div>
                  <div class="val-gate-item" id="gateTaxBox">
                    <span class="gate-title">เลขประจำตัวผู้เสียภาษี (Tax ID Mod 11)</span>
                    <span class="gate-desc" id="gateTaxDesc">-</span>
                  </div>
                  <div class="val-gate-item" id="gateMathBox">
                    <span class="gate-title">กระทบยอดตัวเลข (Financial Math Reconciliation)</span>
                    <span class="gate-desc" id="gateMathDesc">-</span>
                  </div>
                  <div class="val-gate-item" id="gatePolicyBox">
                    <span class="gate-title">กฎระเบียบและเงื่อนไข (University Policy)</span>
                    <span class="gate-desc" id="gatePolicyDesc">-</span>
                  </div>
                </div>

                <!-- Issues List (if any) -->
                <div id="valIssuesContainer" style="display: none; margin-top: 0.7rem; flex-direction: column; gap: 0.4rem;">
                  <div style="font-size: 0.76rem; font-weight: 600; color: #cbd5e1;">รายการข้อผิดพลาดและข้อสังเกต (Issues):</div>
                  <div id="valIssuesList" style="display: flex; flex-direction: column; gap: 0.35rem;"></div>
                </div>
              </div>

              <!-- Dynamic Document Fields Card -->
              <div class="field-card" id="documentFieldsCard">
                <div class="field-card-title">
                  <span>ข้อมูลสำคัญของเอกสาร (Extracted Key Fields)</span>
                  <span id="docTypeTag" class="badge green">ระบุประเภท</span>
                </div>
                <div class="field-grid" id="fieldGrid">
                  <!-- Generated dynamically from result.fields -->
                </div>
              </div>

              <!-- Line Items Table Card -->
              <div class="field-card" id="lineItemsCard">
                <div class="field-card-title">
                  <span>รายการสินค้าและบริการ (Line Items)</span>
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
                    ตรวจสอบยอดถูกต้อง
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

        <!-- Tab 5: PP-ChatOCRv4 Interactive Chat & Knowledge Payload -->
        <div class="tab-content" id="tabChatOCR">
          <div class="chatocr-container">
            <!-- Top Controls: 100% Automated Type-Directed Template Extraction -->
            <div class="chatocr-ctrl-box">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div>
                  <h3 style="font-size: 0.95rem; font-weight: 600; color: #f8fafc; display: flex; align-items: center; gap: 0.4rem;">
                    PP-ChatOCRv4: ระบบสกัดข้อมูลอัตโนมัติตาม Template
                  </h3>
                  <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 3px;">
                    สกัดข้อมูลตาม System Prompt และชุดคำถามของประเภทเอกสารโดยอัตโนมัติ 100% โดยไม่ต้องป้อน Prompt หรือคำถามเอง
                  </p>
                </div>
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                  <span id="chatocrStatusBadge" class="badge green">● PP-ChatOCRv4 Ready</span>
                </div>
              </div>

              <!-- Dedicated File Status & Switcher Bar for PP-ChatOCRv4 Flow -->
              <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 8px; padding: 0.45rem 0.8rem; margin-bottom: 0.75rem; flex-wrap: wrap; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                  <span style="font-size: 0.74rem; color: #cbd5e1; font-weight: 500;">เอกสารที่เลือก:</span>
                  <span id="chatocrCurrentFileBadge" class="badge" style="background: rgba(168, 85, 247, 0.15); color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.3); font-size: 0.76rem;">
                    บิลตัวอย่าง (sample_receipt.png)
                  </span>
                </div>
                <div style="display: flex; gap: 0.4rem;">
                  <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()" style="padding: 0.22rem 0.65rem; font-size: 0.74rem;">
                    เลือกไฟล์สำหรับ PP-ChatOCRv4
                  </button>
                  <button class="btn btn-secondary" onclick="loadSampleReceipt()" style="padding: 0.22rem 0.65rem; font-size: 0.74rem;">
                    ใช้บิลตัวอย่าง
                  </button>
                </div>
              </div>

              <!-- Flow Independence Banner -->
              <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 6px; padding: 0.4rem 0.75rem; font-size: 0.74rem; color: #d8b4fe; display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.65rem;">
                <span><b>โฟลว์อิสระ 100%:</b> อัปโหลดไฟล์แล้วสามารถกด <b>"สกัดข้อมูลอัตโนมัติ"</b> ได้ทันที ไม่ต้องรอ Custom OCR Pipeline</span>
                <span class="badge" style="font-size: 0.68rem; background: rgba(168, 85, 247, 0.2); color: #e9d5ff;">อิสระ ไม่บล็อก</span>
              </div>

              <!-- Automated Controls Row -->
              <div style="display: flex; gap: 0.75rem; align-items: flex-end; flex-wrap: wrap; margin-bottom: 0.6rem;">
                <div style="flex: 1; min-width: 280px;">
                  <label style="font-size: 0.76rem; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 0.3rem;">
                    เลือกประเภทเอกสาร (Document Type Template):
                  </label>
                  <select id="chatocrDocTypeSelect" class="search-input" style="width: 100%; padding: 0.45rem 0.6rem; font-size: 0.82rem;" onchange="onChatOcrDocTypeChange()">
                    <optgroup label="เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)">
                      <option value="principle_approval_request" selected>เอกสารขออนุมัติหลักการ</option>
                      <option value="principle_approval_granted">เอกสารอนุมัติหลักการ</option>
                      <option value="disbursement_approval_request">ขออนุมัติเบิกจ่าย</option>
                      <option value="advance_payment_request_1">แบบเบิกเงินทดรองจ่าย (แบบที่ 1 - สัญญายืมเงิน/เบิก)</option>
                      <option value="advance_payment_request_2">แบบเบิกเงินทดรองจ่าย (แบบที่ 2 - รับเงิน/เคลียร์เงิน)</option>
                      <option value="receipt_substitute">ใบแทนใบเสร็จ / ใบสำคัญรับเงิน</option>
                      <option value="parcel_inspection">ใบตรวจรับพัสดุ</option>
                      <option value="procurement_approval_request">ขออนุมัติจัดหาพัสดุ</option>
                      <option value="procurement_attachment">เอกสารประกอบการขออนุมัติจัดหา</option>
                    </optgroup>
                    <optgroup label="เอกสารประกอบภายนอก (หมวดเสริม)">
                      <option value="general_receipt">ใบเสร็จรับเงิน / ใบกำกับภาษีทั่วไป (ร้านค้า/บริษัท)</option>
                    </optgroup>
                  </select>
                </div>

                <div style="flex-shrink: 0;">
                  <button class="btn" id="btnRunChatOCR" onclick="executeChatOCR()" style="padding: 0.5rem 1.25rem; font-size: 0.85rem; font-weight: 600; background: linear-gradient(135deg, #2563eb, #10b981); box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);">
                    สกัดข้อมูลอัตโนมัติ (PP-ChatOCRv4)
                  </button>
                </div>
              </div>

              <!-- Automated Target Keys Preview Chips -->
              <div style="background: rgba(0, 0, 0, 0.2); border-radius: 6px; padding: 0.5rem 0.75rem; border: 1px solid rgba(255, 255, 255, 0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                  <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">ฟิลด์ที่จะสกัดตาม Template อัตโนมัติ:</span>
                  <span style="font-size: 0.7rem; color: #10b981; font-weight: 500;">System Prompt ผูกตามประเภทเอกสารแล้ว</span>
                </div>
                <div id="chatocrTargetKeysChips" class="preset-chips-group"></div>
              </div>
            </div>

            <!-- Loading Indicator -->
            <div id="chatocrLoading" style="display: none; align-items: center; justify-content: center; gap: 0.75rem; padding: 2.5rem; color: #94a3b8;">
              <div class="spinner" style="display: block; width: 24px; height: 24px;"></div>
              <span id="chatocrLoadingText">PP-ChatOCRv4 กำลังสกัดคำตอบตาม Template และสร้าง Knowledge Payload...</span>
            </div>

            <!-- Empty State -->
            <div id="chatocrEmpty" style="color: var(--text-muted); font-size: 0.85rem; text-align: center; padding: 3rem;">
              กรุณาเลือกไฟล์เอกสาร (หรือกดปุ่ม <b>"ทดสอบบิลตัวอย่าง"</b> ด้านบน) จากนั้นเลือกประเภทเอกสารแล้วกด <b>"สกัดข้อมูลอัตโนมัติ (PP-ChatOCRv4)"</b>
            </div>

            <!-- ChatOCR Results Area -->
            <div id="chatocrResultsArea" style="display: none; flex-direction: column; gap: 0.9rem;">
              <!-- Meta Badge & Actions Header -->
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; background: rgba(0, 0, 0, 0.25); border-radius: 8px; padding: 0.5rem 0.75rem; border: 1px solid var(--border);">
                <div style="display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap;">
                  <span id="chatocrResDocTypeBadge" class="badge green">ใบเสร็จรับเงิน</span>
                  <span id="chatocrResModelBadge" class="badge">PP-ChatOCRv4 + Qwen2.5:3B</span>
                  <span id="chatocrResLatencyBadge" class="badge">- ms</span>
                  <span id="chatocrMathReconcileBadge" class="badge" style="display: none;"></span>
                  <span id="chatocrDbStatusBadge" class="badge" style="font-size: 0.7rem; color: #93c5fd; border-color: rgba(59, 130, 246, 0.4);">PostgreSQL: ตรวจสอบ...</span>
                </div>
                <div style="display: flex; gap: 0.45rem; align-items: center;">
                  <button class="btn btn-secondary" style="padding: 0.28rem 0.65rem; font-size: 0.75rem;" onclick="copyChatOcrAnswersJson()">คัดลอก Q&A JSON</button>
                  <button class="btn" id="btnSaveToPostgres" style="padding: 0.28rem 0.85rem; font-size: 0.76rem; font-weight: 600; background: linear-gradient(135deg, #059669, #10b981); box-shadow: 0 2px 10px rgba(16, 185, 129, 0.35);" onclick="saveToDatabase()">
                    บันทึกลง Database (PostgreSQL)
                  </button>
                </div>
              </div>

              <!-- Database Save Feedback Toast/Banner -->
              <div id="dbSaveToast" style="display: none; padding: 0.65rem 0.95rem; border-radius: 8px; font-size: 0.82rem; line-height: 1.45; transition: all 0.25s ease;"></div>

              <!-- Financial Reconciliation Banner -->
              <div id="chatocrReconcileBanner" style="display: none; padding: 0.65rem 0.95rem; border-radius: 8px; font-size: 0.82rem; line-height: 1.45; transition: all 0.2s ease;"></div>

              <!-- Case Link Selector: Principle-Centric Dossier Linkage -->
              <div id="caseLinkSelectorBox" style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(59, 130, 246, 0.35); border-radius: 8px; padding: 0.75rem 0.95rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem; flex-wrap: wrap; gap: 0.35rem;">
                  <div>
                    <span style="font-size: 0.84rem; font-weight: 600; color: #93c5fd;">การเชื่อมโยงชุดเรื่องเบิกจ่าย (Principle-Centric Case Dossier Linkage)</span>
                    <div style="font-size: 0.69rem; color: var(--text-muted);">เลือกชุดเรื่องที่ต้องการนำเอกสารนี้ไปผูกรวม (ยึดเลขที่ขออนุมัติหลักการเป็น Root Reference)</div>
                  </div>
                  <span id="caseDossierStatusBadge" class="badge" style="font-size: 0.7rem; background: rgba(59, 130, 246, 0.16); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3);">พร้อมเชื่อมโยง</span>
                </div>
                <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
                  <select id="caseSelectDropdown" class="search-input" style="flex: 1; min-width: 270px; font-size: 0.82rem; padding: 0.45rem 0.6rem;" onchange="onCaseSelectChange()">
                    <option value="__new__">[+] สร้างชุดเรื่องใหม่ (New Case Dossier)</option>
                  </select>
                  <button class="btn btn-secondary" type="button" onclick="refreshCaseDropdown()" style="padding: 0.4rem 0.75rem; font-size: 0.75rem;" title="รีเฟรชรายการชุดเรื่อง">รีเฟรช</button>
                </div>
                <div id="aiCaseHintBox" style="display: none; margin-top: 0.5rem; font-size: 0.76rem; padding: 0.4rem 0.65rem; border-radius: 6px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399;"></div>
              </div>

              <!-- Interactive Field Editor & Live Sync Section -->
              <div class="field-editor-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 0.4rem;">
                  <div>
                    <span style="font-size: 0.84rem; font-weight: 600; color: #60a5fa;">แก้ไขข้อมูลก่อนบันทึก (Interactive Field Editor - แก้ไขแล้วซิงค์กับ Embed & Metadata ทันที)</span>
                    <div style="font-size: 0.68rem; color: var(--text-muted);">แก้ไขฟิลด์ด้านล่าง แล้วระบบจะคำนวณสูตรและอัปเดตเนื้อหา Embed Text อัตโนมัติ</div>
                  </div>
                  <button class="btn btn-secondary" style="padding: 0.2rem 0.55rem; font-size: 0.7rem;" onclick="resetFieldsFromOriginalOcr()" title="รีเซ็ตกลับเป็นค่าเริ่มต้นที่สกัดได้จาก OCR">
                    รีเซ็ตค่าเดิมจาก OCR
                  </button>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 0.65rem; padding: 0.35rem 0;">
                  <div>
                    <label style="font-size: 0.72rem; color: #38bdf8; font-weight: 600; display: block; margin-bottom: 0.25rem;">เลขที่เอกสารหลักการต้นเรื่อง (Principle Ref):</label>
                    <input type="text" id="editPrincipleDocNo" class="search-input" style="width: 100%; font-size: 0.82rem; border-color: rgba(56, 189, 248, 0.4);" placeholder="เช่น อว 0602/1234 หรือว่างไว้" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">ยอดรวมทั้งสิ้น (Total Amount - บาท):</label>
                    <input type="number" step="0.01" id="editTotalAmount" class="search-input" style="width: 100%; font-weight: 600; color: #38bdf8; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">รวมก่อนภาษี (Subtotal - บาท):</label>
                    <input type="number" step="0.01" id="editSubtotal" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">ภาษีมูลค่าเพิ่ม (VAT - บาท):</label>
                    <input type="number" step="0.01" id="editVat" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">เลขที่เอกสาร / หนังสือ:</label>
                    <input type="text" id="editDocNo" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">บุคคล / หน่วยงาน / ร้านค้า:</label>
                    <input type="text" id="editVendor" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">วันที่เอกสาร (ISO YYYY-MM-DD):</label>
                    <input type="text" id="editDate" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">เลขประจำตัวผู้เสียภาษี 13 หลัก:</label>
                    <input type="text" id="editTaxId" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                  <div>
                    <label style="font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: block; margin-bottom: 0.25rem;">ชื่อเอกสาร (Document Title):</label>
                    <input type="text" id="editDocTitle" class="search-input" style="width: 100%; font-size: 0.82rem;" oninput="onFormFieldChange()">
                  </div>
                </div>
              </div>

              <!-- 1. Q&A Answers Section -->
              <div class="chatocr-answers-card">
                <div class="chatocr-card-header">
                  <div>
                    <span style="font-weight: 600; font-size: 0.82rem; color: #f1f5f9;">ผลลัพธ์การตอบคำถาม (Extracted Q&A Pairs)</span>
                    <span style="font-size: 0.68rem; color: var(--text-muted); margin-left: 0.4rem;">(สามารถคลิกแก้ไขคำตอบในแต่ละข้อได้โดยตรง และจะซิงค์กับ Embed อัตโนมัติ)</span>
                  </div>
                  <span class="badge" id="chatocrAnswerCountBadge">0 คำตอบ</span>
                </div>
                <div class="chatocr-qa-list" id="chatocrQaList"></div>
              </div>

              <!-- 2. Dual Knowledge Payload Section for Vector DB -->
              <div class="knowledge-payload-container">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                  <h4 style="font-size: 0.85rem; font-weight: 600; color: #38bdf8; display: flex; align-items: center; gap: 0.4rem;">
                    Knowledge Payload สำหรับ Vector DB (ก้อนข้อมูลพร้อมใช้งาน)
                  </h4>
                  <span style="font-size: 0.72rem; color: var(--text-muted);">Embed Dense Text + Metadata Filter</span>
                </div>

                <div class="payload-grid">
                  <!-- Payload 1: embed_text -->
                  <div class="payload-card">
                    <div class="payload-card-header">
                      <div>
                        <span style="font-weight: 600; font-size: 0.78rem; color: #34d399;">Text to Embed (Natural Language Content)</span>
                        <div style="font-size: 0.68rem; color: var(--text-muted);">เนื้อหาสำคัญสรุปให้อ่านเข้าใจ สำหรับนำไปทำ Vector Embedding</div>
                      </div>
                      <button class="btn btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.72rem;" onclick="copyEmbedText()">
                        คัดลอก
                      </button>
                    </div>
                    <div class="code-scroll-container" style="max-height: 240px; background: rgba(0, 0, 0, 0.4); border-radius: 6px; padding: 0.6rem;">
                      <pre id="payloadEmbedTextView" style="margin: 0; font-family: monospace; font-size: 0.75rem; white-space: pre-wrap; color: #e2e8f0;"></pre>
                    </div>
                  </div>

                  <!-- Payload 2: filter_metadata -->
                  <div class="payload-card">
                    <div class="payload-card-header">
                      <div>
                        <span style="font-weight: 600; font-size: 0.78rem; color: #60a5fa;">Filter Metadata (Structured Attributes)</span>
                        <div style="font-size: 0.68rem; color: var(--text-muted);">Attribute สำคัญสำหรับตั้งเงื่อนไข WHERE Filter ใน Vector DB</div>
                      </div>
                      <button class="btn btn-secondary" style="padding: 0.2rem 0.5rem; font-size: 0.72rem;" onclick="copyFilterMetadata()">
                        คัดลอก
                      </button>
                    </div>
                    <div class="code-scroll-container" style="max-height: 240px; background: rgba(0, 0, 0, 0.4); border-radius: 6px; padding: 0.6rem;">
                      <pre id="payloadMetadataView" style="margin: 0; font-family: monospace; font-size: 0.75rem; white-space: pre-wrap; color: #93c5fd;"></pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>
        </div>

        <!-- Tab 6: Principle-Centric Case Dossier Dashboard -->
        <div class="tab-content" id="tabDashboard">
          <div class="dashboard-view">
            <!-- Top Filter & Search Toolbar -->
            <div class="dashboard-toolbar">
              <div style="display: flex; gap: 0.6rem; align-items: center; flex: 1; min-width: 280px; flex-wrap: wrap;">
                <input type="text" id="dashboardSearchInput" class="search-input" style="flex: 1; min-width: 180px; font-size: 0.82rem;" placeholder="ค้นหาเลขที่หลักการ, ชื่อโครงการ, ผู้เบิก, หรือเลขที่บิล..." oninput="filterDashboardCases()">
                <select id="dashboardStatusFilter" class="search-input" style="width: auto; font-size: 0.82rem;" onchange="filterDashboardCases()">
                  <option value="ALL">สถานะทั้งหมด</option>
                  <option value="APPROVED_PRINCIPLE">อนุมัติหลักการแล้ว</option>
                  <option value="PENDING_APPROVAL">รอดำเนินการเบิกจ่าย</option>
                  <option value="RECONCILED">เบิกจ่ายเรียบร้อย (ในงบ)</option>
                  <option value="OVER_BUDGET">เกินวงเงินหลักการ</option>
                </select>
              </div>
              <div style="display: flex; gap: 0.45rem; align-items: center;">
                <button class="btn btn-secondary" onclick="loadDashboardData()" style="padding: 0.35rem 0.75rem; font-size: 0.78rem;">
                  รีเฟรชข้อมูล
                </button>
              </div>
            </div>

            <!-- 4 KPI Cards -->
            <div class="kpi-grid">
              <div class="kpi-card">
                <span class="kpi-title">จำนวนชุดเรื่อง (Dossiers)</span>
                <span class="kpi-value" id="kpiTotalCases" style="color: #60a5fa;">0</span>
                <span class="kpi-sub">ชุดเรื่องเบิกจ่ายทั้งหมด</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-title">เอกสารทั้งหมดที่จัดเก็บ</span>
                <span class="kpi-value" id="kpiTotalDocs" style="color: #a78bfa;">0</span>
                <span class="kpi-sub">ไฟล์ใน PostgreSQL</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-title">งบประมาณหลักการรวม</span>
                <span class="kpi-value" id="kpiApprovedBudget" style="color: #38bdf8;">฿0.00</span>
                <span class="kpi-sub">เพดานวงเงินที่อนุมัติ</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-title">ยอดเบิกจริง / คงเหลือ</span>
                <span class="kpi-value" id="kpiActualSpent" style="color: #34d399;">฿0.00</span>
                <span class="kpi-sub" id="kpiRemainingBudget" style="color: #94a3b8;">คงเหลือ ฿0.00</span>
              </div>
            </div>

            <!-- Dossier Cases List Container -->
            <div id="dashboardCasesList" style="display: flex; flex-direction: column; gap: 0.85rem;"></div>

            <!-- Empty State -->
            <div id="dashboardEmptyState" style="display: none; color: var(--text-muted); text-align: center; padding: 3rem; background: rgba(0,0,0,0.2); border-radius: 8px;">
              ยังไม่มีข้อมูลชุดเรื่องในระบบ หรือไม่พบรายการที่ค้นหา
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
    let currentUploadedFile = null;
    let isUsingSample = false;
    let activeTab = 'blocks';
    let currentFlowMode = 'custom';
    let customPipelineController = null;

    function setAppFlow(flow) {
      currentFlowMode = flow;
      const btnCustom = document.getElementById('flowBtnCustom');
      const btnChatOCR = document.getElementById('flowBtnChatOCR');
      const uploadBtn = document.getElementById('btnUploadFile');
      const sampleBtn = document.getElementById('btnLoadSample');
      const deskewLabel = document.getElementById('deskewCheckLabel');
      const flowBadge = document.getElementById('activeFlowBadge');

      if (flow === 'chatocr') {
        if (btnCustom) btnCustom.classList.remove('active');
        if (btnChatOCR) btnChatOCR.classList.add('active');

        if (uploadBtn) uploadBtn.innerHTML = 'เลือกไฟล์สำหรับ PP-ChatOCRv4';
        if (sampleBtn) sampleBtn.innerHTML = 'บิลตัวอย่าง (PP-ChatOCRv4)';
        if (deskewLabel) deskewLabel.style.display = 'none';

        // Abort background custom pipeline if it was still running
        if (customPipelineController) {
          customPipelineController.abort();
          customPipelineController = null;
        }

        switchTab('chatocr');
        setStatus(currentUploadedFile ? `โหมด PP-ChatOCRv4: พร้อมสกัดข้อมูล "${currentUploadedFile.name}" (กดปุ่มด้านล่าง)` : 'โหมด PP-ChatOCRv4 สำเร็จรูป (เลือกไฟล์แล้วกดปุ่มสกัดได้ทันที)', false);
      } else {
        if (btnCustom) btnCustom.classList.add('active');
        if (btnChatOCR) btnChatOCR.classList.remove('active');

        if (uploadBtn) uploadBtn.innerHTML = 'เลือกไฟล์ภาพหรือ PDF';
        if (sampleBtn) sampleBtn.innerHTML = 'ทดสอบบิลตัวอย่าง (Sample)';
        if (deskewLabel) deskewLabel.style.display = 'inline-flex';

        switchTab('blocks');
        setStatus('โหมด Custom Pipeline พร้อมใช้งาน', false);
      }
    }

    document.getElementById('fileInput').addEventListener('change', function(e) {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });

    function handleFileSelection(file) {
      if (!file) return;
      currentUploadedFile = file;
      isUsingSample = false;

      // Update filename badges
      const chatFileBadge = document.getElementById('chatocrCurrentFileBadge');
      if (chatFileBadge) {
        chatFileBadge.innerText = file.name;
      }

      // Instant Client-Side Image Preview (0ms delay)
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
          const docImage = document.getElementById('docImage');
          docImage.src = e.target.result;
          docImage.style.display = 'block';
          document.getElementById('emptyState').style.display = 'none';
          document.getElementById('pageInfo').innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
          zoomReset();
        };
        reader.readAsDataURL(file);
      } else {
        document.getElementById('pageInfo').innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      }

      // Route according to active flow
      if (activeTab === 'chatocr' || currentFlowMode === 'chatocr') {
        // Dedicated PP-ChatOCRv4 Flow: Abort any custom pipeline request & DO NOT trigger heavy custom pipeline OCR!
        if (customPipelineController) {
          customPipelineController.abort();
          customPipelineController = null;
        }
        setStatus(`โหลด "${file.name}" สำหรับ PP-ChatOCRv4 เรียบร้อย (กดปุ่มสกัดด้านล่างได้ทันที)`, false);
        const chatSelect = document.getElementById('chatocrDocTypeSelect');
        if (chatSelect && chatSelect.value === 'general_receipt') {
          chatSelect.value = 'principle_approval_request';
          onChatOcrDocTypeChange();
        }
      } else {
        // Custom Pipeline Flow: Run full OCR extraction
        uploadAndProcess(file);
      }
    }

    async function loadSampleReceipt() {
      currentUploadedFile = null;
      isUsingSample = true;

      // Update PP-ChatOCR filename badge
      const chatFileBadge = document.getElementById('chatocrCurrentFileBadge');
      if (chatFileBadge) {
        chatFileBadge.innerText = 'บิลตัวอย่าง (sample_receipt.png)';
      }

      // Auto-select general receipt for sample receipt
      const chatSelect = document.getElementById('chatocrDocTypeSelect');
      if (chatSelect) {
        chatSelect.value = 'general_receipt';
        onChatOcrDocTypeChange();
      }
      const llmSelect = document.getElementById('llmDocTypeSelect');
      if (llmSelect) {
        llmSelect.value = 'general_receipt';
      }

      // If user is on PP-ChatOCR tab or mode, load sample preview immediately without running heavy custom OCR pipeline
      if (activeTab === 'chatocr' || currentFlowMode === 'chatocr') {
        if (customPipelineController) {
          customPipelineController.abort();
          customPipelineController = null;
        }
        setStatus('กำลังโหลดภาพบิลตัวอย่าง...', true);
        try {
          const res = await fetch('/api/sample/preview');
          const data = await res.json();
          const docImage = document.getElementById('docImage');
          docImage.src = 'data:image/png;base64,' + data.image_base64;
          docImage.style.display = 'block';
          document.getElementById('emptyState').style.display = 'none';
          document.getElementById('pageInfo').innerText = `${data.width}x${data.height} px`;
          zoomReset();
          setStatus('โหลดบิลตัวอย่างสำหรับ PP-ChatOCRv4 พร้อมสกัดแล้ว (กดปุ่มสกัดได้ทันที)', false);
        } catch (err) {
          setStatus('โหลดบิลตัวอย่างพร้อมสกัดแล้ว', false);
        }
        return;
      }

      setStatus('กำลังโหลดบิลตัวอย่าง (Custom Pipeline)...', true);
      try {
        const res = await fetch('/api/sample');
        const data = await res.json();
        displayResults(data);
        setStatus('สกัดข้อความภาษาไทยสำเร็จ! (Custom Pipeline)', false);
      } catch (err) {
        setStatus('เกิดข้อผิดพลาด: ' + err.message, false);
      }
    }

    async function uploadAndProcess(file) {
      if (customPipelineController) {
        customPipelineController.abort();
      }
      customPipelineController = new AbortController();

      setStatus('กำลังประมวลผล OCR ภาษาไทย (Custom Pipeline)...', true);
      currentUploadedFile = file;
      isUsingSample = false;
      const formData = new FormData();
      formData.append('file', file);
      formData.append('auto_deskew', document.getElementById('deskewCheck').checked);

      try {
        const res = await fetch('/api/extract', {
          method: 'POST',
          body: formData,
          signal: customPipelineController.signal
        });
        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.error || 'Extract failed');
        }
        const data = await res.json();
        customPipelineController = null;
        displayResults(data);
        setStatus('สกัดข้อความภาษาไทยสำเร็จ! (Custom Pipeline)', false);
      } catch (err) {
        if (err.name === 'AbortError') {
          console.log('Custom Pipeline aborted by user / flow switch');
          return;
        }
        customPipelineController = null;
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
      activeTab = tabId;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      const flowBadge = document.getElementById('activeFlowBadge');
      const deskewLabel = document.getElementById('deskewCheckLabel');
      const uploadBtn = document.getElementById('btnUploadFile');
      const sampleBtn = document.getElementById('btnLoadSample');
      const btnCustom = document.getElementById('flowBtnCustom');
      const btnChatOCR = document.getElementById('flowBtnChatOCR');

      if (tabId === 'blocks') {
        document.querySelector('.tab-btn:nth-child(1)').classList.add('active');
        document.getElementById('tabBlocks').classList.add('active');
      } else if (tabId === 'markdown') {
        document.querySelector('.tab-btn:nth-child(2)').classList.add('active');
        document.getElementById('tabMarkdown').classList.add('active');
      } else if (tabId === 'extraction') {
        document.getElementById('tabBtnExtraction').classList.add('active');
        document.getElementById('tabExtraction').classList.add('active');
      } else if (tabId === 'chatocr') {
        document.getElementById('tabBtnChatOCR').classList.add('active');
        document.getElementById('tabChatOCR').classList.add('active');
      } else if (tabId === 'json') {
        document.getElementById('tabBtnJson').classList.add('active');
        document.getElementById('tabJson').classList.add('active');
      } else if (tabId === 'dashboard') {
        document.getElementById('tabBtnDashboard').classList.add('active');
        document.getElementById('tabDashboard').classList.add('active');
        loadDashboardData();
      }

      // Update flow indicator badge & toolbar
      if (tabId === 'chatocr') {
        currentFlowMode = 'chatocr';
        if (btnCustom) btnCustom.classList.remove('active');
        if (btnChatOCR) btnChatOCR.classList.add('active');
        if (flowBadge) {
          flowBadge.innerHTML = 'โฟลว์: PP-ChatOCRv4 (PaddleX)';
          flowBadge.style.background = 'rgba(168, 85, 247, 0.2)';
          flowBadge.style.color = '#d8b4fe';
          flowBadge.style.borderColor = 'rgba(168, 85, 247, 0.4)';
        }
        if (uploadBtn) uploadBtn.innerHTML = 'เลือกไฟล์สำหรับ PP-ChatOCRv4';
        if (sampleBtn) sampleBtn.innerHTML = 'บิลตัวอย่าง (PP-ChatOCRv4)';
        if (deskewLabel) deskewLabel.style.display = 'none';

        // Abort background custom pipeline if it was running
        if (customPipelineController) {
          customPipelineController.abort();
          customPipelineController = null;
        }

        setStatus(currentUploadedFile ? `พร้อมสกัดข้อมูลด้วย PP-ChatOCRv4: "${currentUploadedFile.name}" (กดปุ่มสกัดด้านล่าง)` : 'PP-ChatOCRv4 พร้อมใช้งาน (เลือกไฟล์แล้วกดปุ่มสกัด)', false);
      } else if (tabId === 'dashboard') {
        if (flowBadge) {
          flowBadge.innerHTML = 'แดชบอร์ดติดตามเรื่อง (Dossier Traceability)';
          flowBadge.style.background = 'rgba(59, 130, 246, 0.2)';
          flowBadge.style.color = '#93c5fd';
          flowBadge.style.borderColor = 'rgba(59, 130, 246, 0.4)';
        }
        setStatus('แดชบอร์ดชุดเรื่องเบิกจ่าย (Dossiers) แสดงรายการเอกสารและการกระทบยอดงบประมาณ', false);
      } else {
        currentFlowMode = 'custom';
        if (btnCustom) btnCustom.classList.add('active');
        if (btnChatOCR) btnChatOCR.classList.remove('active');
        if (flowBadge) {
          flowBadge.innerHTML = 'โฟลว์: Custom Pipeline';
          flowBadge.style.background = 'rgba(37, 99, 235, 0.15)';
          flowBadge.style.color = '#93c5fd';
          flowBadge.style.borderColor = 'rgba(37, 99, 235, 0.3)';
        }
        if (uploadBtn) uploadBtn.innerHTML = 'เลือกไฟล์ภาพหรือ PDF';
        if (sampleBtn) sampleBtn.innerHTML = 'ทดสอบบิลตัวอย่าง (Sample)';
        if (deskewLabel) deskewLabel.style.display = 'inline-flex';
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
          badge.innerText = '● Ollama พร้อมใช้งาน (GPU)';
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
          badge.innerText = 'Ollama Offline (ใช้ Real Snapshot)';
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
          const officialGroup = document.createElement('optgroup');
          officialGroup.label = 'เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)';
          const suppGroup = document.createElement('optgroup');
          suppGroup.label = 'เอกสารประกอบภายนอก (หมวดเสริม)';

          const bmOfficialGroup = document.createElement('optgroup');
          bmOfficialGroup.label = 'เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)';
          const bmSuppGroup = document.createElement('optgroup');
          bmSuppGroup.label = 'เอกสารประกอบภายนอก (หมวดเสริม)';

          data.supported_document_types.forEach((dt, idx) => {
            const opt = document.createElement('option');
            opt.value = dt.id;
            opt.innerText = dt.title;
            if (idx === 0) opt.selected = true;
            if (dt.id === 'general_receipt') {
              suppGroup.appendChild(opt);
            } else {
              officialGroup.appendChild(opt);
            }

            if (bmDocSelect) {
              const bmOpt = document.createElement('option');
              bmOpt.value = dt.id;
              bmOpt.innerText = dt.title;
              if (dt.id === 'general_receipt' || idx === 0) bmOpt.selected = true;
              if (dt.id === 'general_receipt') {
                bmSuppGroup.appendChild(bmOpt);
              } else {
                bmOfficialGroup.appendChild(bmOpt);
              }
            }
          });

          docSelect.appendChild(officialGroup);
          if (suppGroup.children.length > 0) docSelect.appendChild(suppGroup);

          if (bmDocSelect) {
            bmDocSelect.appendChild(bmOfficialGroup);
            if (bmSuppGroup.children.length > 0) bmDocSelect.appendChild(bmSuppGroup);
          }
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
      btn.innerText = 'กำลังประมวลผล...';
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
      document.getElementById('resLatencyBadge').innerText = `${(result.latency_ms / 1000).toFixed(2)}s`;

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
          badgeEl.innerText = '[PASSED] ผ่านการตรวจสอบทั้งหมด';
          valStatusBadge.className = 'badge green';
          valStatusBadge.innerText = 'ผ่านเกณฑ์';
        } else if (val.status === 'WARNING') {
          badgeEl.className = 'badge';
          badgeEl.style.color = '#fbbf24';
          badgeEl.style.borderColor = 'rgba(245, 158, 11, 0.4)';
          badgeEl.innerText = '[WARNING] ผ่านแบบมีข้อสังเกต';
          valStatusBadge.className = 'badge';
          valStatusBadge.style.color = '#fbbf24';
          valStatusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
          valStatusBadge.innerText = 'มีข้อสังเกต';
        } else {
          badgeEl.className = 'badge';
          badgeEl.style.color = '#f87171';
          badgeEl.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          badgeEl.innerText = '[ERROR] ตรวจพบข้อผิดพลาด';
          valStatusBadge.className = 'badge';
          valStatusBadge.style.color = '#f87171';
          valStatusBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          valStatusBadge.innerText = 'มีข้อผิดพลาด';
        }

        // 1. Date Gate
        const dateDesc = document.getElementById('gateDateDesc');
        if (val.date_report && val.date_report.is_valid) {
          dateDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">${val.date_report.iso_date}</span> <small style="color: var(--text-muted);">(${val.date_report.thai_formatted})</small>`;
        } else if (val.date_report && val.date_report.raw_date) {
          dateDesc.innerHTML = `<span style="color: #fbbf24; font-weight: 600;">${val.date_report.raw_date}</span>`;
        } else {
          dateDesc.innerHTML = `<span style="color: var(--text-muted);">- ไม่ระบุวันที่ -</span>`;
        }

        // 2. Tax ID Gate
        const taxDesc = document.getElementById('gateTaxDesc');
        if (val.tax_id_report && val.tax_id_report.is_valid) {
          taxDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">${val.tax_id_report.formatted_id}</span> <small style="color: #94a3b8;">(Mod 11 Checksum ผ่าน)</small>`;
        } else if (val.tax_id_report && val.tax_id_report.raw_id) {
          taxDesc.innerHTML = `<span style="color: #f87171; font-weight: 600;">${val.tax_id_report.raw_id}</span> <small style="color: #fca5a5;">(${val.tax_id_report.error_message || 'ไม่ผ่าน'})</small>`;
        } else {
          taxDesc.innerHTML = `<span style="color: var(--text-muted);">- ไม่มีระบุในเอกสาร -</span>`;
        }

        // 3. Math Gate
        const mathDesc = document.getElementById('gateMathDesc');
        if (val.math_report && val.math_report.is_balanced) {
          mathDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">[ตรงกัน] ยอดเงินสอดคล้อง</span> <small style="color: var(--text-muted); display: block;">${val.math_report.details || ''}</small>`;
        } else if (val.math_report) {
          mathDesc.innerHTML = `<span style="color: #f87171; font-weight: 600;">[ไม่ตรงกัน] ยอดเงินคลาดเคลื่อน</span> <small style="color: #fca5a5; display: block;">${val.math_report.details || ''}</small>`;
        } else {
          mathDesc.innerHTML = `<span style="color: var(--text-muted);">- ไม่มียอดเงินที่ต้องคำนวณ -</span>`;
        }

        // 4. University Policy Gate
        const policyDesc = document.getElementById('gatePolicyDesc');
        const policyIssues = (val.issues || []).filter(i => i.code === 'MISSING_REQUIRED_FIELD' || i.code === 'PETTY_CASH_EXCEEDS_LIMIT' || i.code === 'EMPTY_EXPENSE_ITEMS');
        if (policyIssues.length === 0) {
          policyDesc.innerHTML = `<span style="color: #34d399; font-weight: 600;">ครบถ้วนตามระเบียบมหาวิทยาลัย</span>`;
        } else {
          const hasErr = policyIssues.some(i => i.severity === 'ERROR');
          policyDesc.innerHTML = `<span style="color: ${hasErr ? '#f87171' : '#fbbf24'}; font-weight: 600;">${hasErr ? '[ไม่ครบ] ขาดข้อมูลจำเป็น' : '[แจ้งเตือน] มีเงื่อนไขเตือน'} (${policyIssues.length} จุด)</span>`;
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
              <span style="font-weight: 700; font-size: 0.8rem; color: ${issue.severity === 'ERROR' ? '#f87171' : '#fbbf24'};">[${issue.severity}]</span>
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
        doc_no: "เลขที่เอกสาร",
        doc_date: "วันที่ทำเอกสาร",
        title: "เรื่อง",
        requester: "ผู้ทำการเบิก / หน่วยงาน",
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
          mathBadge.innerText = `[ตรวจสอบถูกต้อง] Subtotal + VAT = Total: ฿${formatCurrency(actualTotal)}`;
        } else {
          mathBadge.style.display = 'inline-flex';
          mathBadge.className = 'badge';
          mathBadge.style.color = '#f87171';
          mathBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          mathBadge.innerText = `[ยอดไม่ตรงกัน] ระบุ: ฿${formatCurrency(actualTotal)} (ต่างจาก Subtotal + VAT: ฿${formatCurrency(expectedTotal)})`;
        }
      } else if (total) {
        mathBadge.style.display = 'inline-flex';
        mathBadge.className = 'badge green';
        mathBadge.innerText = `ยอดสุทธิระบุในเอกสาร: ฿${formatCurrency(total)}`;
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

    // Initialize LLM status, ChatOCR templates and Database status on load
    window.addEventListener('DOMContentLoaded', () => {
      loadLLMStatus();
      loadChatOcrTemplates();
      checkDatabaseStatus();
      loadPrincipleCasesForDropdown();
      loadDashboardData();
    });

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

    /* PP-ChatOCRv4 Automated Client Logic (Component 6) */
    let lastChatOcrResult = null;
    let chatocrTemplatesMap = {};

    async function loadChatOcrTemplates() {
      try {
        const res = await fetch('/api/chatocr/templates');
        const templates = await res.json();
        const select = document.getElementById('chatocrDocTypeSelect');
        const prevVal = select ? select.value : null;
        select.innerHTML = '';
        chatocrTemplatesMap = {};

        const officialGroup = document.createElement('optgroup');
        officialGroup.label = 'เอกสารเบิกจ่ายราชการ (9 ประเภทหลัก)';
        const suppGroup = document.createElement('optgroup');
        suppGroup.label = 'เอกสารประกอบภายนอก (หมวดเสริม)';

        templates.forEach((tpl) => {
          chatocrTemplatesMap[tpl.id] = tpl;
          const opt = document.createElement('option');
          opt.value = tpl.id;
          opt.innerText = tpl.title;
          if (tpl.group === 'supplementary') {
            suppGroup.appendChild(opt);
          } else {
            officialGroup.appendChild(opt);
          }
        });

        select.appendChild(officialGroup);
        if (suppGroup.children.length > 0) {
          select.appendChild(suppGroup);
        }

        if (prevVal && chatocrTemplatesMap[prevVal]) {
          select.value = prevVal;
        } else if (isUsingSample) {
          select.value = 'general_receipt';
        } else {
          select.value = 'principle_approval_request';
        }

        onChatOcrDocTypeChange();
      } catch (err) {
        console.warn('Could not load chatocr templates:', err);
      }
    }

    function onChatOcrDocTypeChange() {
      const select = document.getElementById('chatocrDocTypeSelect');
      const docType = select ? select.value : 'principle_approval_request';
      const container = document.getElementById('chatocrTargetKeysChips');
      if (!container) return;
      container.innerHTML = '';

      const tpl = chatocrTemplatesMap[docType];
      const keys = tpl ? tpl.keys : [
        "เลขที่เอกสารหรือเลขที่หนังสือ",
        "วันที่ทำเอกสาร",
        "เรื่อง",
        "ผู้ทำการเบิกหรือหน่วยงานที่ขอ",
        "รายละเอียดค่าใช้จ่าย",
        "ยอดรวมเงินงบประมาณที่ขออนุมัติ"
      ];

      keys.forEach((k, idx) => {
        const chip = document.createElement('span');
        chip.className = 'preset-chip';
        chip.style.cursor = 'default';
        chip.style.background = 'rgba(59, 130, 246, 0.12)';
        chip.style.borderColor = 'rgba(59, 130, 246, 0.3)';
        chip.style.color = '#93c5fd';
        chip.innerText = `${idx + 1}. ${k}`;
        container.appendChild(chip);
      });
    }

    async function executeChatOCR() {
      // Abort background custom pipeline immediately so resources are freed
      if (customPipelineController) {
        customPipelineController.abort();
        customPipelineController = null;
      }

      if (!currentUploadedFile && !isUsingSample) {
        isUsingSample = true;
      }

      const docType = document.getElementById('chatocrDocTypeSelect').value || 'general_receipt';

      const btn = document.getElementById('btnRunChatOCR');
      const loader = document.getElementById('chatocrLoading');
      const emptyState = document.getElementById('chatocrEmpty');
      const resultsArea = document.getElementById('chatocrResultsArea');

      btn.disabled = true;
      loader.style.display = 'flex';
      emptyState.style.display = 'none';
      resultsArea.style.display = 'none';
      setStatus('PP-ChatOCRv4 กำลังสกัดคำตอบตาม Template...', true);

      const formData = new FormData();
      if (currentUploadedFile) {
        formData.append('file', currentUploadedFile);
        formData.append('use_sample', false);
      } else {
        formData.append('use_sample', true);
      }

      formData.append('document_type', docType);

      try {
        const res = await fetch('/api/chatocr/chat', {
          method: 'POST',
          body: formData
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.error || 'ChatOCR request failed');
        }

        const data = await res.json();
        lastChatOcrResult = data;
        renderChatOcrResults(data);
        setStatus('สกัดข้อมูลด้วย PP-ChatOCRv4 สำเร็จ!', false);
      } catch (err) {
        alert('เกิดข้อผิดพลาดในการสกัดข้อมูล PP-ChatOCRv4: ' + err.message);
        emptyState.style.display = 'block';
        setStatus('เกิดข้อผิดพลาดในการสกัดข้อมูล PP-ChatOCRv4', false);
      } finally {
        btn.disabled = false;
        loader.style.display = 'none';
      }
    }

    let lastChatOcrResult = null;
    let rawOriginalChatOcrResult = null;
    let currentChatOcrResult = null;

    function escapeHtml(str) {
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    function renderChatOcrResults(data) {
      const resultsArea = document.getElementById('chatocrResultsArea');
      const emptyState = document.getElementById('chatocrEmpty');
      emptyState.style.display = 'none';
      resultsArea.style.display = 'flex';

      // Save deep copies of original and working copy
      lastChatOcrResult = data;
      rawOriginalChatOcrResult = JSON.parse(JSON.stringify(data));
      currentChatOcrResult = JSON.parse(JSON.stringify(data));

      // Update left stage document preview if server returned base64 image
      if (data.image_base64) {
        const docImage = document.getElementById('docImage');
        docImage.src = 'data:image/png;base64,' + data.image_base64;
        docImage.style.display = 'block';
        document.getElementById('emptyState').style.display = 'none';
        zoomReset();
      }

      // Meta badges
      const meta = (currentChatOcrResult.knowledge_payload && currentChatOcrResult.knowledge_payload.filter_metadata) || {};
      document.getElementById('chatocrResDocTypeBadge').innerText = meta.document_type_name || data.document_type_name || data.document_title || data.document_type || 'receipt';
      document.getElementById('chatocrResModelBadge').innerText = `${data.model_info?.engine || 'PP-ChatOCRv4'} + ${data.model_info?.llm_model || 'Qwen2.5:3B'}`;
      document.getElementById('chatocrResLatencyBadge').innerText = `${(data.latency_ms / 1000).toFixed(2)}s`;

      // Badge in Tab Header
      const badge = document.getElementById('chatocrBadge');
      badge.style.display = 'inline-flex';
      badge.innerText = 'ตอบแล้ว';

      // Populate interactive editor inputs
      populateFieldEditor(currentChatOcrResult);

      // Render Q&A answers with editable textareas
      renderQaPairs(currentChatOcrResult.chat_answers || {});

      // Trigger initial real-time sync & reconciliation calculation
      onFormFieldChange();

      // Check available cases and trigger smart candidate matching
      loadPrincipleCasesForDropdown(meta.principle_doc_no, meta.total_amount, meta.vendor_or_requester);

      // Check and update database status
      checkDatabaseStatus();
    }

    function populateFieldEditor(data) {
      const meta = (data.knowledge_payload && data.knowledge_payload.filter_metadata) || {};
      const elPrincipleNo = document.getElementById('editPrincipleDocNo');
      const elTotal = document.getElementById('editTotalAmount');
      const elSubtotal = document.getElementById('editSubtotal');
      const elVat = document.getElementById('editVat');
      const elDocNo = document.getElementById('editDocNo');
      const elVendor = document.getElementById('editVendor');
      const elDate = document.getElementById('editDate');
      const elTaxId = document.getElementById('editTaxId');
      const elDocTitle = document.getElementById('editDocTitle');

      if (elPrincipleNo) elPrincipleNo.value = meta.principle_doc_no || data.principle_doc_no || '';
      if (elTotal) elTotal.value = meta.total_amount != null ? meta.total_amount : '';
      if (elSubtotal) elSubtotal.value = meta.subtotal != null ? meta.subtotal : '';
      if (elVat) elVat.value = meta.vat != null ? meta.vat : '';
      if (elDocNo) elDocNo.value = meta.document_no || meta.doc_no || '';
      if (elVendor) elVendor.value = meta.vendor || meta.vendor_or_requester || '';
      if (elDate) elDate.value = meta.date || meta.doc_date_iso || '';
      if (elTaxId) elTaxId.value = meta.tax_id || meta.vendor_tax_id || '';
      if (elDocTitle) elDocTitle.value = meta.document_title || data.filename || 'เอกสารการเงิน';
    }

    function renderQaPairs(answers) {
      const qaList = document.getElementById('chatocrQaList');
      if (!qaList) return;
      qaList.innerHTML = '';
      const keys = Object.keys(answers || {});
      document.getElementById('chatocrAnswerCountBadge').innerText = `${keys.length} คำตอบ`;

      keys.forEach((q, idx) => {
        const a = answers[q] || '';
        const isCrossCheck = q.includes('Cross-check') || q.includes('ตรวจสอบ') || q.includes('ความถูกต้อง');
        const item = document.createElement('div');
        item.className = 'chatocr-qa-item' + (isCrossCheck ? ' crosscheck-item' : '');
        item.innerHTML = `
          <div class="chatocr-q-title">
            <span>[${idx + 1}]</span>
            <span style="${isCrossCheck ? 'color: #38bdf8; font-weight: 600;' : ''}">${escapeHtml(q)}</span>
            ${isCrossCheck ? '<span class="badge" style="font-size: 0.65rem; background: rgba(56, 189, 248, 0.2); color: #38bdf8; margin-left: auto;">Financial Cross-Check</span>' : ''}
          </div>
          <div style="margin-top: 0.35rem;">
            <textarea class="chatocr-qa-input" data-key="${escapeHtml(q)}" oninput="onQaAnswerChange(this)" rows="${a.length > 80 ? 3 : 2}" placeholder="- ไม่มีระบุ -">${escapeHtml(a)}</textarea>
          </div>
        `;
        qaList.appendChild(item);
      });
    }

    function onQaAnswerChange(textarea) {
      if (!currentChatOcrResult) return;
      const key = textarea.getAttribute('data-key');
      const val = textarea.value;
      if (!currentChatOcrResult.chat_answers) currentChatOcrResult.chat_answers = {};
      currentChatOcrResult.chat_answers[key] = val;

      // Auto-sync into form field if it's total amount or subtotal or doc number
      if (key.includes('ยอดเงินรวม') || key.includes('ยอดรวมทั้งสิ้น') || key.includes('วงเงินงบประมาณ')) {
        const numMatch = val.replace(/,/g, '').match(/([0-9]+(?:[.][0-9]+)?)/);
        if (numMatch) {
          const parsed = parseFloat(numMatch[1]);
          if (!isNaN(parsed) && parsed > 0) {
            document.getElementById('editTotalAmount').value = parsed;
          }
        }
      }

      onFormFieldChange();
    }

    function onFormFieldChange() {
      if (!currentChatOcrResult) return;

      if (!currentChatOcrResult.knowledge_payload) {
        currentChatOcrResult.knowledge_payload = { filter_metadata: {}, embed_text: '' };
      }
      if (!currentChatOcrResult.knowledge_payload.filter_metadata) {
        currentChatOcrResult.knowledge_payload.filter_metadata = {};
      }

      const meta = currentChatOcrResult.knowledge_payload.filter_metadata;

      // 1. Read input values from DOM
      const principleDocNo = document.getElementById('editPrincipleDocNo')?.value.trim() || null;
      const rawTotal = document.getElementById('editTotalAmount')?.value.trim();
      const rawSubtotal = document.getElementById('editSubtotal')?.value.trim();
      const rawVat = document.getElementById('editVat')?.value.trim();
      const docNo = document.getElementById('editDocNo')?.value.trim() || null;
      const vendor = document.getElementById('editVendor')?.value.trim() || null;
      const dateStr = document.getElementById('editDate')?.value.trim() || null;
      const taxId = document.getElementById('editTaxId')?.value.trim() || null;
      const docTitle = document.getElementById('editDocTitle')?.value.trim() || null;

      const totalAmount = rawTotal !== '' && !isNaN(parseFloat(rawTotal)) ? parseFloat(rawTotal) : null;
      const subtotal = rawSubtotal !== '' && !isNaN(parseFloat(rawSubtotal)) ? parseFloat(rawSubtotal) : null;
      const vat = rawVat !== '' && !isNaN(parseFloat(rawVat)) ? parseFloat(rawVat) : null;

      // 2. Synchronize into filter_metadata
      meta.principle_doc_no = principleDocNo;
      meta.total_amount = totalAmount;
      meta.subtotal = subtotal;
      meta.vat = vat;
      meta.doc_no = docNo;
      meta.document_no = docNo;
      meta.vendor_or_requester = vendor;
      meta.vendor = vendor;
      meta.doc_date_iso = dateStr;
      meta.date = dateStr;
      meta.vendor_tax_id = taxId;
      meta.tax_id = taxId;
      meta.document_title = docTitle || meta.document_title || currentChatOcrResult.filename || 'เอกสารการเงิน';

      // 3. Dynamic Real-Time Math Reconciliation
      let reconcile = {
        status: 'unverified',
        is_balanced: false,
        diff: 0.0,
        calculated_total: totalAmount,
        details: 'ไม่มีข้อมูลกระทบยอด'
      };

      // Check for formulas in chat answers (e.g. "600 x 2" or "600*2 = 1200")
      let formulaFound = null;
      const answers = currentChatOcrResult.chat_answers || {};
      for (const [k, v] of Object.entries(answers)) {
        if (!v) continue;
        const cleaned = v.replace(/,/g, '');
        const m = cleaned.match(/([0-9]+(?:[.][0-9]+)?)[ \t]*(?:[*]|x|คูณ)[ \t]*([0-9]+(?:[.][0-9]+)?)/i);
        if (m) {
          const n1 = parseFloat(m[1]);
          const n2 = parseFloat(m[2]);
          if (!isNaN(n1) && !isNaN(n2)) {
            formulaFound = { n1, n2, result: Math.round((n1 * n2) * 100) / 100 };
            break;
          }
        }
      }

      if (formulaFound && totalAmount !== null) {
        const diff = Math.round(Math.abs(formulaFound.result - totalAmount) * 100) / 100;
        if (diff <= 0.05) {
          reconcile = {
            status: 'passed',
            is_balanced: true,
            diff: 0.0,
            calculated_total: totalAmount,
            details: `ยอดเงินตรงกันสมบูรณ์: คำนวณสูตร ${formulaFound.n1.toLocaleString('th-TH', {minimumFractionDigits: 2})} x ${formulaFound.n2} = ${totalAmount.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท`
          };
        } else {
          reconcile = {
            status: 'discrepancy',
            is_balanced: false,
            diff: diff,
            calculated_total: formulaFound.result,
            details: `ตรวจพบยอดเงินไม่ตรงกัน (Discrepancy): คำนวณสูตร ${formulaFound.n1.toLocaleString('th-TH', {minimumFractionDigits: 2})} x ${formulaFound.n2} = ${formulaFound.result.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท แต่ระบุยอดรวม ${totalAmount.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท (ต่างกัน ${diff.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท)`
          };
        }
      } else if (subtotal !== null && vat !== null && totalAmount !== null) {
        const expected = Math.round((subtotal + vat) * 100) / 100;
        const diff = Math.round(Math.abs(expected - totalAmount) * 100) / 100;
        if (diff <= 0.05) {
          reconcile = {
            status: 'passed',
            is_balanced: true,
            diff: 0.0,
            calculated_total: totalAmount,
            details: `รวมก่อนภาษี (${subtotal.toLocaleString('th-TH', {minimumFractionDigits: 2})}) + VAT (${vat.toLocaleString('th-TH', {minimumFractionDigits: 2})}) เท่ากับยอดเงินรวมทั้งสิ้น (${totalAmount.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท) ถูกต้องสมบูรณ์`
          };
        } else {
          reconcile = {
            status: 'discrepancy',
            is_balanced: false,
            diff: diff,
            calculated_total: expected,
            details: `รวมก่อนภาษี (${subtotal.toLocaleString('th-TH', {minimumFractionDigits: 2})}) + VAT (${vat.toLocaleString('th-TH', {minimumFractionDigits: 2})}) = ${expected.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท ไม่ตรงกับยอดเงินรวมทั้งสิ้น ${totalAmount.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท (ต่างกัน ${diff.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท)`
          };
        }
      } else if (totalAmount !== null) {
        reconcile = {
          status: 'unverified',
          is_balanced: false,
          diff: 0.0,
          calculated_total: totalAmount,
          details: `มียอดเงินระบุ ${totalAmount.toLocaleString('th-TH', {minimumFractionDigits: 2})} บาท (ยังไม่พบรายการย่อยสำหรับกระทบยอด)`
        };
      }

      meta.math_reconciliation = reconcile;

      // 4. Update Reconciliation UI Banner & Badge
      const recBadge = document.getElementById('chatocrMathReconcileBadge');
      const recBanner = document.getElementById('chatocrReconcileBanner');

      if (recBadge && recBanner) {
        if (reconcile.status === 'passed') {
          recBadge.style.display = 'inline-flex';
          recBadge.className = 'badge green';
          recBadge.style.background = '';
          recBadge.style.color = '';
          recBadge.style.border = '';
          recBadge.innerText = '[ถูกต้อง] ตรวจสอบยอดเงินถูกต้อง';

          recBanner.style.display = 'block';
          recBanner.style.background = 'rgba(16, 185, 129, 0.12)';
          recBanner.style.border = '1px solid rgba(16, 185, 129, 0.35)';
          recBanner.style.color = '#34d399';
          recBanner.innerHTML = `<b>[ตรวจสอบถูกต้อง] ตรวจสอบความถูกต้องของยอดเงิน (Reconciliation Passed):</b> ${reconcile.details}`;
        } else if (reconcile.status === 'discrepancy') {
          recBadge.style.display = 'inline-flex';
          recBadge.className = 'badge';
          recBadge.style.background = 'rgba(239, 68, 68, 0.2)';
          recBadge.style.color = '#ef4444';
          recBadge.style.border = '1px solid rgba(239, 68, 68, 0.4)';
          recBadge.innerText = '[ไม่ตรงกัน] ตรวจพบยอดเงินคลาดเคลื่อน (Discrepancy)';

          recBanner.style.display = 'block';
          recBanner.style.background = 'rgba(239, 68, 68, 0.12)';
          recBanner.style.border = '1px solid rgba(239, 68, 68, 0.35)';
          recBanner.style.color = '#f87171';
          recBanner.innerHTML = `<b>[ยอดไม่ตรงกัน] ตรวจพบความคลาดเคลื่อนของยอดเงิน (Math Discrepancy):</b> ${reconcile.details}`;
        } else {
          recBadge.style.display = 'none';
          recBanner.style.display = 'none';
        }
      }

      // 5. Reconstruct Embed Text Live
      const newEmbedText = buildEmbedText(meta, currentChatOcrResult.chat_answers);
      currentChatOcrResult.knowledge_payload.embed_text = newEmbedText;

      // 6. Update Viewers in UI
      const embedView = document.getElementById('payloadEmbedTextView');
      const metaView = document.getElementById('payloadMetadataView');
      if (embedView) embedView.innerText = newEmbedText;
      if (metaView) metaView.innerText = JSON.stringify(meta, null, 2);

      // Keep lastChatOcrResult in sync
      lastChatOcrResult = currentChatOcrResult;
    }

    function buildEmbedText(meta, chatAnswers) {
      const statusThaiMap = {
        passed: 'ตรวจสอบถูกต้อง (Reconciled)',
        discrepancy: 'พบยอดเงินไม่ตรงกัน (Discrepancy)',
        verified_by_llm: 'ยืนยันผ่าน AI Cross-Check',
        unverified: 'ยังไม่ได้ตรวจสอบ (Unverified)'
      };
      const rec = meta.math_reconciliation || {};
      const statusThai = statusThaiMap[rec.status] || rec.status || 'ยังไม่ได้ตรวจสอบ';
      const recDetails = rec.details || 'ไม่มีข้อมูลกระทบยอด';

      const lines = [
        '# เอกสารการเงิน: ' + (meta.document_title || 'เอกสารการเงิน'),
        '- **ชื่อเอกสาร (Document Title):** ' + (meta.document_title || 'เอกสารการเงิน'),
        '- **ประเภทเอกสาร:** ' + (meta.document_type_name || meta.document_type || 'ไม่ระบุ'),
        '- **เลขที่เอกสาร:** ' + (meta.doc_no || meta.document_no || 'ไม่ระบุ'),
        '- **เลขที่เอกสารหลักการต้นเรื่อง (Principle Ref):** ' + (meta.principle_doc_no || 'ไม่ระบุ'),
        '- **บุคคล/หน่วยงาน/ร้านค้า:** ' + (meta.vendor_or_requester || meta.vendor || 'ไม่ระบุ'),
        '- **วันที่เอกสาร:** ' + (meta.doc_date_iso || meta.date || 'ไม่ระบุ'),
        '- **ยอดเงินรวม:** ' + (meta.total_amount != null ? meta.total_amount.toLocaleString('th-TH', {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' บาท' : 'ไม่ระบุ'),
        '- **การตรวจสอบความถูกต้องของยอดเงิน (Reconciliation):** ' + statusThai + ' - ' + recDetails
      ];

      if (meta.subtotal != null) {
        lines.push('- **ยอดรวมก่อนภาษี (Subtotal):** ' + meta.subtotal.toLocaleString('th-TH', {minimumFractionDigits: 2}) + ' บาท');
      }
      if (meta.vat != null) {
        lines.push('- **ภาษีมูลค่าเพิ่ม (VAT):** ' + meta.vat.toLocaleString('th-TH', {minimumFractionDigits: 2}) + ' บาท');
      }
      if (meta.tax_id || meta.vendor_tax_id) {
        lines.push('- **เลขประจำตัวผู้เสียภาษี:** ' + (meta.tax_id || meta.vendor_tax_id));
      }

      lines.push('');
      lines.push('## ข้อมูลที่สกัดได้ตาม Template (PP-ChatOCRv4):');
      for (const [q, a] of Object.entries(chatAnswers || {})) {
        lines.push('- **' + q + ':** ' + (a || 'ไม่มีระบุ'));
      }

      return lines.join('\n');
    }

    function resetFieldsFromOriginalOcr() {
      if (!rawOriginalChatOcrResult) {
        alert('ยังไม่มีข้อมูลต้นฉบับจาก OCR');
        return;
      }
      currentChatOcrResult = JSON.parse(JSON.stringify(rawOriginalChatOcrResult));
      populateFieldEditor(currentChatOcrResult);
      renderQaPairs(currentChatOcrResult.chat_answers || {});
      onFormFieldChange();
      setStatus('รีเซ็ตฟิลด์ข้อมูลกลับสู่ค่าต้นฉบับจาก OCR เรียบร้อยแล้ว', false);
    }

    async function checkDatabaseStatus() {
      const headerBadge = document.getElementById('headerDbBadge');
      const chatocrBadge = document.getElementById('chatocrDbStatusBadge');

      try {
        const res = await fetch('/api/db/status');
        if (!res.ok) throw new Error('DB status request failed');
        const data = await res.json();

        if (data.connected) {
          const docCount = data.documents_count || 0;
          if (headerBadge) {
            headerBadge.innerText = `PostgreSQL: พร้อมใช้งาน (${docCount} เอกสาร)`;
            headerBadge.className = 'badge green';
            headerBadge.style.color = '#34d399';
            headerBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          }
          if (chatocrBadge) {
            chatocrBadge.innerText = `PostgreSQL: เชื่อมต่อสำเร็จ (${docCount} เอกสาร)`;
            chatocrBadge.style.color = '#34d399';
            chatocrBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          }
        } else {
          if (headerBadge) {
            headerBadge.innerText = 'PostgreSQL: ขัดข้อง';
            headerBadge.className = 'badge red';
            headerBadge.style.color = '#f87171';
            headerBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          }
          if (chatocrBadge) {
            chatocrBadge.innerText = 'PostgreSQL: ไม่พร้อมใช้งาน';
            chatocrBadge.style.color = '#f87171';
            chatocrBadge.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          }
        }
      } catch (err) {
        if (headerBadge) {
          headerBadge.innerText = 'PostgreSQL: เชื่อมต่อไม่ได้';
          headerBadge.className = 'badge red';
        }
        if (chatocrBadge) {
          chatocrBadge.innerText = 'PostgreSQL: เชื่อมต่อไม่ได้';
        }
      }
    }

    async function saveToDatabase() {
      if (!currentChatOcrResult) {
        alert('กรุณาสกัดข้อมูลเอกสารด้วย PP-ChatOCRv4 ก่อนทำการบันทึกลง Database');
        return;
      }

      const btn = document.getElementById('btnSaveToPostgres');
      const toast = document.getElementById('dbSaveToast');
      const origText = btn ? btn.innerHTML : '';

      try {
        if (btn) {
          btn.disabled = true;
          btn.innerHTML = 'กำลังบันทึกลง Database...';
        }

        const caseSelectVal = document.getElementById('caseSelectDropdown')?.value;
        const isNewCase = (!caseSelectVal || caseSelectVal === '__new__');
        const targetRecordId = isNewCase ? null : caseSelectVal;
        const meta = currentChatOcrResult.knowledge_payload?.filter_metadata || {};

        const payload = {
          filename: currentUploadedFile ? currentUploadedFile.name : (currentChatOcrResult.filename || 'sample_receipt.png'),
          document_type: currentChatOcrResult.document_type || document.getElementById('chatocrDocTypeSelect')?.value || 'general_receipt',
          metadata: meta,
          embed_text: currentChatOcrResult.knowledge_payload?.embed_text || '',
          chat_answers: currentChatOcrResult.chat_answers || {},
          math_reconciliation: meta.math_reconciliation || {},
          target_record_id: targetRecordId,
          is_new_case: isNewCase,
          principle_doc_no: meta.principle_doc_no || null,
          case_title: meta.document_title || null
        };

        const res = await fetch('/api/chatocr/save_to_db', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok || !data.success) {
          throw new Error(data.error || 'บันทึกข้อมูลไม่สำเร็จ');
        }

        if (toast) {
          toast.style.display = 'block';
          toast.style.background = 'rgba(16, 185, 129, 0.15)';
          toast.style.border = '1px solid rgba(16, 185, 129, 0.4)';
          toast.style.color = '#34d399';
          toast.innerHTML = `
            <b>บันทึกข้อมูลลงฐานข้อมูล PostgreSQL สำเร็จเรียบร้อย!</b><br>
            <span style="font-size: 0.76rem; color: #a7f3d0;">
              ชุดเรื่อง (Case Record ID): <code style="background: rgba(0,0,0,0.3); padding: 0.1rem 0.35rem; border-radius: 4px;">${data.record_id}</code> | 
              รหัสเอกสาร (Doc ID): <code style="background: rgba(0,0,0,0.3); padding: 0.1rem 0.35rem; border-radius: 4px;">${data.document_id}</code>
            </span><br>
            <span style="font-size: 0.72rem; color: #94a3b8;">
              สถานะ: บันทึกลงตาราง disbursement_records, documents (JSONB extracted_data), expense_items, work_logs ครบถ้วน
            </span>
          `;
        }

        setStatus('บันทึกข้อมูลลงฐานข้อมูล PostgreSQL เรียบร้อยแล้ว!', false);
        await checkDatabaseStatus();
        await loadPrincipleCasesForDropdown(meta.principle_doc_no, meta.total_amount, meta.vendor_or_requester);
        await loadDashboardData();
      } catch (err) {
        if (toast) {
          toast.style.display = 'block';
          toast.style.background = 'rgba(239, 68, 68, 0.15)';
          toast.style.border = '1px solid rgba(239, 68, 68, 0.4)';
          toast.style.color = '#f87171';
          toast.innerHTML = `<b>เกิดข้อผิดพลาดในการบันทึกลง Database:</b> ${escapeHtml(err.message)}`;
        }
        alert('เกิดข้อผิดพลาดในการบันทึก: ' + err.message);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = origText;
        }
      }
    }

    /* Principle-Centric Case Linking & Dossier Dashboard Functions */
    let availablePrincipleCases = [];
    let dashboardDataCache = null;

    async function loadPrincipleCasesForDropdown(preferredPrincipleNo, amount, vendor) {
      const select = document.getElementById('caseSelectDropdown');
      const badge = document.getElementById('caseDossierStatusBadge');
      const hint = document.getElementById('aiCaseHintBox');
      if (!select) return;

      try {
        const res = await fetch('/api/db/cases');
        if (!res.ok) throw new Error('Failed to fetch cases');
        availablePrincipleCases = await res.json();

        select.innerHTML = '<option value="__new__">[+] สร้างชุดเรื่องใหม่ (New Case Dossier)</option>';
        availablePrincipleCases.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.id;
          opt.innerText = `[ชุดเรื่อง] ${c.principle_doc_no} : ${c.title} (งบ ฿${c.approved_amount.toLocaleString()} | เบิกแล้ว ฿${c.actual_expense.toLocaleString()})`;
          select.appendChild(opt);
        });

        // Trigger Smart Case Candidate Matching
        const docType = currentChatOcrResult?.document_type || document.getElementById('chatocrDocTypeSelect')?.value || '';
        if (availablePrincipleCases.length > 0 && docType !== 'principle_approval_request') {
          const matchParams = new URLSearchParams();
          if (preferredPrincipleNo) matchParams.append('principle_doc_no', preferredPrincipleNo);
          if (amount) matchParams.append('amount', amount);
          if (vendor) matchParams.append('vendor', vendor);

          const matchRes = await fetch(`/api/db/match_case?${matchParams.toString()}`);
          if (matchRes.ok) {
            const matchData = await matchRes.json();
            if (matchData.matched_case_id) {
              select.value = matchData.matched_case_id;
              if (badge) {
                badge.innerText = `เชื่อมโยงกับ ${matchData.principle_doc_no || 'ชุดเรื่อง'}`;
                badge.style.color = '#34d399';
              }
              if (hint) {
                hint.style.display = 'block';
                hint.innerHTML = `<b>AI แนะนำ (${Math.round(matchData.confidence * 100)}%):</b> ${escapeHtml(matchData.reason)}`;
              }
              return;
            }
          }
        }

        if (docType === 'principle_approval_request') {
          select.value = '__new__';
          if (badge) {
            badge.innerText = 'เอกสารหลักการ (สร้างชุดเรื่องใหม่)';
            badge.style.color = '#93c5fd';
          }
          if (hint) hint.style.display = 'none';
        } else {
          if (hint) hint.style.display = 'none';
          if (badge) {
            badge.innerText = select.value === '__new__' ? 'สร้างชุดเรื่องใหม่' : 'ผูกกับชุดเรื่องที่มี';
          }
        }
      } catch (err) {
        console.warn('Error loading principle cases:', err);
      }
    }

    function onCaseSelectChange() {
      const select = document.getElementById('caseSelectDropdown');
      const badge = document.getElementById('caseDossierStatusBadge');
      const hint = document.getElementById('aiCaseHintBox');
      if (!select) return;

      if (select.value === '__new__') {
        if (badge) {
          badge.innerText = 'สร้างชุดเรื่องใหม่';
          badge.style.color = '#93c5fd';
        }
        if (hint) hint.style.display = 'none';
      } else {
        const found = availablePrincipleCases.find(c => c.id === select.value);
        if (badge) {
          badge.innerText = found ? `ผูกกับ ${found.principle_doc_no}` : 'ผูกกับชุดเรื่อง';
          badge.style.color = '#34d399';
        }
      }
    }

    async function refreshCaseDropdown() {
      const meta = currentChatOcrResult?.knowledge_payload?.filter_metadata || {};
      await loadPrincipleCasesForDropdown(meta.principle_doc_no, meta.total_amount, meta.vendor_or_requester);
    }

    async function loadDashboardData() {
      try {
        const res = await fetch('/api/db/dashboard');
        if (!res.ok) throw new Error('Dashboard data request failed');
        const data = await res.json();
        dashboardDataCache = data;

        // Update KPI Cards
        const sum = data.summary || {};
        const kpiCases = document.getElementById('kpiTotalCases');
        const kpiDocs = document.getElementById('kpiTotalDocs');
        const kpiAppr = document.getElementById('kpiApprovedBudget');
        const kpiActual = document.getElementById('kpiActualSpent');
        const kpiRem = document.getElementById('kpiRemainingBudget');

        if (kpiCases) kpiCases.innerText = (sum.total_cases || 0).toLocaleString();
        if (kpiDocs) kpiDocs.innerText = (sum.total_documents || 0).toLocaleString();
        if (kpiAppr) kpiAppr.innerText = `฿${(sum.total_approved_budget || 0).toLocaleString('th-TH', {minimumFractionDigits: 2})}`;
        if (kpiActual) kpiActual.innerText = `฿${(sum.total_actual_expense || 0).toLocaleString('th-TH', {minimumFractionDigits: 2})}`;
        if (kpiRem) kpiRem.innerText = `คงเหลือ ฿${(sum.total_remaining_budget || 0).toLocaleString('th-TH', {minimumFractionDigits: 2})}`;

        // Tab Badge
        const tabBadge = document.getElementById('dashboardCountBadge');
        if (tabBadge) tabBadge.innerText = `${sum.total_cases || 0} เรื่อง`;

        renderDashboardCases(data.cases || []);
      } catch (err) {
        console.warn('Could not load dashboard data:', err);
      }
    }

    function filterDashboardCases() {
      if (!dashboardDataCache || !dashboardDataCache.cases) return;
      const q = (document.getElementById('dashboardSearchInput')?.value || '').toLowerCase().trim();
      const status = document.getElementById('dashboardStatusFilter')?.value || 'ALL';

      const filtered = dashboardDataCache.cases.filter(c => {
        const matchStatus = (status === 'ALL' || c.status === status);
        if (!matchStatus) return false;

        if (!q) return true;
        const inPno = (c.principle_doc_no || '').toLowerCase().includes(q);
        const inTitle = (c.title || '').toLowerCase().includes(q);
        const inReceiver = (c.receiver_name || '').toLowerCase().includes(q);
        const inDocs = (c.documents || []).some(d => 
          (d.doc_no || '').toLowerCase().includes(q) ||
          (d.filename || '').toLowerCase().includes(q) ||
          (d.party_name || '').toLowerCase().includes(q)
        );
        return inPno || inTitle || inReceiver || inDocs;
      });

      renderDashboardCases(filtered);
    }

    function renderDashboardCases(cases) {
      const container = document.getElementById('dashboardCasesList');
      const emptyState = document.getElementById('dashboardEmptyState');
      if (!container) return;

      container.innerHTML = '';
      if (!cases || cases.length === 0) {
        if (emptyState) emptyState.style.display = 'block';
        return;
      }
      if (emptyState) emptyState.style.display = 'none';

      cases.forEach((c, idx) => {
        const card = document.createElement('div');
        card.className = 'case-card';

        // Budget calculations
        const approved = c.approved_amount || 0;
        const actual = c.actual_expense || 0;
        const remaining = c.remaining_amount != null ? c.remaining_amount : (approved - actual);
        const pct = approved > 0 ? Math.min(100, Math.round((actual / approved) * 100)) : (actual > 0 ? 100 : 0);
        const isOverBudget = actual > approved && approved > 0;
        const barColor = isOverBudget ? 'linear-gradient(90deg, #ef4444, #f87171)' : 'linear-gradient(90deg, #3b82f6, #10b981)';

        // Status badge
        let statusBadgeHtml = '';
        if (c.status === 'APPROVED_PRINCIPLE') {
          statusBadgeHtml = '<span class="badge" style="background: rgba(59, 130, 246, 0.15); color: #93c5fd; border-color: rgba(59, 130, 246, 0.35);">อนุมัติหลักการแล้ว</span>';
        } else if (c.status === 'RECONCILED') {
          statusBadgeHtml = '<span class="badge green">เบิกจ่ายเรียบร้อย (ในงบ)</span>';
        } else if (c.status === 'OVER_BUDGET') {
          statusBadgeHtml = '<span class="badge" style="background: rgba(239, 68, 68, 0.18); color: #f87171; border-color: rgba(239, 68, 68, 0.4);">เกินวงเงินหลักการ</span>';
        } else {
          statusBadgeHtml = '<span class="badge" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.35);">รอดำเนินการเบิกจ่าย</span>';
        }

        // Checklist HTML
        const chk = c.checklist || {};
        const chkPrinciple = chk.has_principle ? '<span class="checklist-pill ok">[1] ขออนุมัติหลักการ</span>' : '<span class="checklist-pill missing">[1] ขออนุมัติหลักการ</span>';
        const chkProcurement = chk.has_procurement ? '<span class="checklist-pill ok">[2] จัดหา/จัดจ้าง</span>' : '<span class="checklist-pill missing">[2] จัดหา/จัดจ้าง</span>';
        const chkReceipt = chk.has_receipt ? '<span class="checklist-pill ok">[3] ใบเสร็จ/ใบสำคัญ</span>' : '<span class="checklist-pill missing">[3] ใบเสร็จ/ใบสำคัญ</span>';
        const chkDisbursement = chk.has_disbursement ? '<span class="checklist-pill ok">[4] ขออนุมัติเบิกจ่าย</span>' : '<span class="checklist-pill missing">[4] ขออนุมัติเบิกจ่าย</span>';

        // Child documents rows
        let docsRowsHtml = '';
        (c.documents || []).forEach((d, docIdx) => {
          const recStatusHtml = d.reconciliation_status === 'passed'
            ? '<span class="badge green" style="font-size: 0.65rem;">ตรงกัน</span>'
            : (d.reconciliation_status === 'discrepancy'
              ? '<span class="badge" style="background: rgba(239,68,68,0.2); color: #f87171; font-size: 0.65rem;">คลาดเคลื่อน</span>'
              : '<span class="badge" style="font-size: 0.65rem; color: #94a3b8;">ยังไม่ตรวจ</span>');

          docsRowsHtml += `
            <tr>
              <td style="color: var(--text-dim); text-align: center;">${docIdx + 1}</td>
              <td><span class="badge" style="font-size: 0.67rem; background: rgba(255,255,255,0.06);">${escapeHtml(d.document_type_name || d.document_type)}</span></td>
              <td style="font-weight: 500; color: #93c5fd;">${escapeHtml(d.doc_no || '-')}</td>
              <td>${escapeHtml(d.doc_date_iso || '-')}</td>
              <td>${escapeHtml(d.party_name || '-')}</td>
              <td style="text-align: right; font-weight: 600; color: #f1f5f9;">฿${(d.total_amount || 0).toLocaleString('th-TH', {minimumFractionDigits: 2})}</td>
              <td style="text-align: center;">${recStatusHtml}</td>
              <td style="text-align: center;">
                <button class="btn btn-secondary" style="padding: 0.15rem 0.45rem; font-size: 0.68rem;" onclick="openDocDetailModal('${c.record_id}', '${d.document_id}')">
                  ดูข้อมูล (Detail)
                </button>
              </td>
            </tr>
          `;
        });

        card.innerHTML = `
          <div class="case-header">
            <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
              <span class="case-pno-badge">หลักการ: ${escapeHtml(c.principle_doc_no)}</span>
              <h3 style="font-size: 0.92rem; font-weight: 600; color: #f8fafc;">${escapeHtml(c.title)}</h3>
            </div>
            <div style="display: flex; align-items: center; gap: 0.45rem;">
              ${statusBadgeHtml}
              <span class="badge" style="font-size: 0.72rem; color: var(--text-muted);">${(c.documents || []).length} เอกสารในชุด</span>
            </div>
          </div>

          <!-- Budget Progress Bar -->
          <div style="background: rgba(0,0,0,0.25); padding: 0.55rem 0.75rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
            <div style="display: flex; justify-content: space-between; font-size: 0.74rem; margin-bottom: 0.35rem;">
              <span style="color: var(--text-muted);">
                เบิกจ่ายแล้ว: <b style="color: ${isOverBudget ? '#f87171' : '#34d399'};">฿${actual.toLocaleString('th-TH', {minimumFractionDigits: 2})}</b> / วงเงินหลักการ: <b>฿${approved.toLocaleString('th-TH', {minimumFractionDigits: 2})}</b> (${pct}%)
              </span>
              <span style="color: ${remaining >= 0 ? '#94a3b8' : '#f87171'}; font-weight: 500;">
                ${remaining >= 0 ? 'คงเหลือเบิกได้: ฿' + remaining.toLocaleString('th-TH', {minimumFractionDigits: 2}) : 'เกินวงเงิน: ฿' + Math.abs(remaining).toLocaleString('th-TH', {minimumFractionDigits: 2})}
              </span>
            </div>
            <div class="progress-bar-wrap">
              <div class="progress-bar-fill" style="width: ${pct}%; background: ${barColor};"></div>
            </div>
          </div>

          <!-- Document Traceability Checklist -->
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.4rem; padding-top: 0.1rem;">
            <div style="font-size: 0.72rem; color: var(--text-muted); font-weight: 600;">ความครบถ้วนของเอกสาร (Traceability Checklist):</div>
            <div class="checklist-row">
              ${chkPrinciple}
              ${chkProcurement}
              ${chkReceipt}
              ${chkDisbursement}
            </div>
          </div>

          <!-- Attached Documents Table -->
          <div style="overflow-x: auto;">
            <table class="case-doc-table">
              <thead>
                <tr>
                  <th style="width: 36px; text-align: center;">#</th>
                  <th>ประเภทเอกสาร</th>
                  <th>เลขที่เอกสาร</th>
                  <th>วันที่</th>
                  <th>ร้านค้า / ผู้เบิก</th>
                  <th style="text-align: right;">จำนวนเงิน</th>
                  <th style="text-align: center;">กระทบยอด</th>
                  <th style="text-align: center;">การจัดการ</th>
                </tr>
              </thead>
              <tbody>
                ${docsRowsHtml || '<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 1rem;">ยังไม่มีเอกสารแนบในชุดเรื่องนี้</td></tr>'}
              </tbody>
            </table>
          </div>
        `;

        container.appendChild(card);
      });
    }

    function openDocDetailModal(caseId, docId) {
      if (!dashboardDataCache || !dashboardDataCache.cases) return;
      const targetCase = dashboardDataCache.cases.find(c => c.record_id === caseId);
      if (!targetCase) return;
      const doc = (targetCase.documents || []).find(d => d.document_id === docId);
      if (!doc) return;

      document.getElementById('docDetailModalTitle').innerText = `รายละเอียด: ${doc.filename} [${doc.document_type_name || doc.document_type}]`;

      const header = document.getElementById('docDetailMetaHeader');
      header.innerHTML = `
        <span class="case-pno-badge">หลักการ: ${escapeHtml(targetCase.principle_doc_no)}</span>
        <span class="badge" style="color: #93c5fd;">เลขที่: ${escapeHtml(doc.doc_no || '-')}</span>
        <span class="badge">วันที่: ${escapeHtml(doc.doc_date_iso || '-')}</span>
        <span class="badge green">ยอดเงิน: ฿${(doc.total_amount || 0).toLocaleString('th-TH', {minimumFractionDigits: 2})}</span>
        <span class="badge">${escapeHtml(doc.party_name || '-')}</span>
      `;

      const raw = doc.extracted_data || {};
      const embedText = raw.embed_text || '';
      document.getElementById('docDetailEmbedText').innerText = embedText || 'ไม่มี Embed Text';
      document.getElementById('docDetailJsonView').innerText = JSON.stringify(raw, null, 2);

      document.getElementById('docDetailModal').classList.add('active');
    }

    function closeDocDetailModal() {
      document.getElementById('docDetailModal').classList.remove('active');
    }

    function copyChatOcrAnswersJson() {
      if (!lastChatOcrResult || !lastChatOcrResult.chat_answers) return;
      navigator.clipboard.writeText(JSON.stringify(lastChatOcrResult.chat_answers, null, 2))
        .then(() => alert('คัดลอก Q&A JSON ลง Clipboard แล้ว!'))
        .catch(err => alert('ไม่สามารถคัดลอกได้: ' + err));
    }

    function copyEmbedText() {
      if (!lastChatOcrResult || !lastChatOcrResult.knowledge_payload?.embed_text) return;
      navigator.clipboard.writeText(lastChatOcrResult.knowledge_payload.embed_text)
        .then(() => alert('คัดลอก Text to Embed สำหรับ Vector DB เรียบร้อยแล้ว!'))
        .catch(err => alert('ไม่สามารถคัดลอกได้: ' + err));
    }

    function copyFilterMetadata() {
      if (!lastChatOcrResult || !lastChatOcrResult.knowledge_payload?.filter_metadata) return;
      navigator.clipboard.writeText(JSON.stringify(lastChatOcrResult.knowledge_payload.filter_metadata, null, 2))
        .then(() => alert('คัดลอก Filter Metadata JSON เรียบร้อยแล้ว!'))
        .catch(err => alert('ไม่สามารถคัดลอกได้: ' + err));
    }
  </script>

  <!-- Model Benchmark Modal Dialog (Component 5) -->
  <div class="modal-overlay" id="benchmarkModal">
    <div class="modal-card">
      <div class="modal-header">
        <h3>เปรียบเทียบโมเดล AI สกัดข้อมูลทางการเงิน (LLM Benchmarking)</h3>
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
              เริ่มการทดสอบ (Run Benchmark)
            </button>
          </div>
        </div>

        <div id="bmLoading" style="display: none; align-items: center; justify-content: center; gap: 0.75rem; padding: 2rem; color: #94a3b8;">
          <div class="spinner" style="display: block; width: 24px; height: 24px;"></div>
          <span>กำลังรันการทดสอบและวิเคราะห์ผลเชิงสถิติ... (โปรดรอสักครู่)</span>
        </div>

        <div id="bmResultsArea" style="display: none; flex-direction: column; gap: 0.8rem;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 600; color: #e2e8f0;">ตารางเปรียบเทียบผลลัพธ์ (Markdown Format)</span>
            <button class="btn btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="copyBenchmarkMarkdown()">
              คัดลอก Markdown สำหรับสไลด์ / เล่มรายงาน
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
  <!-- Document Detail Modal Dialog -->
  <div class="modal-overlay" id="docDetailModal">
    <div class="modal-card" style="max-width: 820px; width: 92%;">
      <div class="modal-header">
        <h3 id="docDetailModalTitle">รายละเอียดเอกสารในชุดเรื่อง</h3>
        <button class="btn-icon" onclick="closeDocDetailModal()" style="font-size: 1.1rem;">✕</button>
      </div>
      <div class="modal-body" style="display: flex; flex-direction: column; gap: 0.85rem; max-height: 70vh; overflow-y: auto;">
        <div id="docDetailMetaHeader" style="display: flex; gap: 0.4rem; flex-wrap: wrap; align-items: center;"></div>
        <div>
          <span style="font-size: 0.78rem; font-weight: 600; color: #34d399;">Natural Language Embed Text (Vector Knowledge):</span>
          <div class="code-scroll-container" style="max-height: 200px; background: rgba(0,0,0,0.4); border-radius: 6px; padding: 0.65rem; margin-top: 0.3rem;">
            <pre id="docDetailEmbedText" style="margin: 0; font-size: 0.75rem; white-space: pre-wrap; color: #e2e8f0;"></pre>
          </div>
        </div>
        <div>
          <span style="font-size: 0.78rem; font-weight: 600; color: #60a5fa;">Raw Extracted JSON Data (PostgreSQL JSONB):</span>
          <div class="code-scroll-container" style="max-height: 220px; background: rgba(0,0,0,0.4); border-radius: 6px; padding: 0.65rem; margin-top: 0.3rem;">
            <pre id="docDetailJsonView" style="margin: 0; font-size: 0.75rem; color: #93c5fd; white-space: pre-wrap;"></pre>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-secondary" onclick="closeDocDetailModal()">ปิดหน้าต่าง</button>
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


@app.get("/api/sample/preview")
async def get_sample_preview():
    """Returns sample image preview immediately without running heavy OCR engine."""
    sample_path = ROOT_DIR / "tests" / "output" / "sample_receipt.png"
    if not sample_path.exists():
        from tests.test_components_1_2 import create_sample_receipt_image
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        img = create_sample_receipt_image()
        img.save(sample_path)
    else:
        img = Image.open(sample_path)

    return {
        "filename": "sample_receipt.png",
        "width": img.width,
        "height": img.height,
        "image_base64": pil_to_base64(img)
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


@app.get(
    "/api/chatocr/templates",
    tags=["PP-ChatOCRv4 (Component 6)"],
    summary="Get Predefined Extraction Templates for 10 Document Types"
)
async def api_chatocr_templates():
    """Returns metadata and target keys for all 10 automated document templates."""
    return chat_ocr_engine.get_supported_templates()


@app.post(
    "/api/chatocr/chat",
    tags=["PP-ChatOCRv4 (Component 6)"],
    summary="Automated Document Extraction & Knowledge Payload with PP-ChatOCRv4"
)
async def api_chatocr_chat(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    questions: Optional[str] = Form(None),
    document_type: str = Form("general_receipt"),
    custom_prompt: Optional[str] = Form(None)
):
    """
    Component 6: Automated Visual Document Extraction & Knowledge Payload Generator.
    Uses PaddleX PP-ChatOCRv4 with Thai OCR (th_PP-OCRv5) and Local LLM (qwen2.5:3b).
    Automatically applies type-directed system prompts and predefined target keys.
    Generates dual Knowledge Payload (text to embed + filter metadata).
    """
    import time
    try:
        target_path: Optional[Path] = None

        if use_sample or (file is None):
            target_path = ROOT_DIR / "tests" / "output" / "sample_receipt.png"
            if not target_path.exists():
                return JSONResponse({"error": "Sample receipt not found on server"}, status_code=404)
        else:
            raw_bytes = await file.read()
            filename = file.filename or "upload.png"
            suffix = Path(filename).suffix.lower()
            upload_dir = ROOT_DIR / "scratch" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            if suffix == ".pdf" or raw_bytes.startswith(b"%PDF"):
                pages = ingestor.load_document(raw_bytes)
                if not pages:
                    return JSONResponse({"error": "Failed to parse PDF document pages"}, status_code=400)
                temp_path = upload_dir / f"pdf_page_1_{int(time.time() * 1000)}.png"
                pages[0].image.save(temp_path, format="PNG")
                target_path = temp_path
            else:
                temp_path = upload_dir / f"upload_{int(time.time() * 1000)}{suffix or '.png'}"
                with open(temp_path, "wb") as f:
                    f.write(raw_bytes)
                target_path = temp_path

        # Parse questions list if provided, else None (triggers automatic template keys)
        q_list = None
        if questions and questions.strip():
            q_list = []
            for line in questions.replace("\r", "").split("\n"):
                for part in line.split(","):
                    q_clean = part.strip()
                    if q_clean and q_clean not in q_list:
                        q_list.append(q_clean)

        res = await run_in_threadpool(
            chat_ocr_engine.chat_and_extract,
            image_path=target_path,
            questions=q_list,
            document_type=document_type,
            custom_prompt=custom_prompt
        )

        # Attach image_base64 so frontend left viewer displays the document
        if target_path and target_path.exists():
            import base64
            img_bytes = target_path.read_bytes()
            res["image_base64"] = base64.b64encode(img_bytes).decode("utf-8")

        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


from src.db import (
    check_db_status,
    save_disbursement_document,
    get_recent_saved_documents,
    get_available_principle_cases,
    find_smart_case_match,
    get_dossier_dashboard_data,
)


class ChatOcrSaveDbPayload(BaseModel):
    filename: Optional[str] = "document.png"
    document_type: Optional[str] = "general_receipt"
    metadata: Optional[dict] = {}
    embed_text: Optional[str] = ""
    chat_answers: Optional[dict] = {}
    math_reconciliation: Optional[dict] = {}
    target_record_id: Optional[str] = None
    is_new_case: Optional[bool] = False
    principle_doc_no: Optional[str] = None
    case_title: Optional[str] = None


@app.get("/api/db/status", tags=["PostgreSQL Database"])
async def api_db_status():
    """Check connectivity to local PostgreSQL container (kxcvbnm/expense-reimbursement-postgres)."""
    return await run_in_threadpool(check_db_status)


@app.get("/api/db/cases", tags=["PostgreSQL Database"])
async def api_db_cases():
    """Fetch available principle cases / dossiers for document linking."""
    return await run_in_threadpool(get_available_principle_cases)


@app.get("/api/db/match_case", tags=["PostgreSQL Database"])
async def api_db_match_case(
    amount: Optional[float] = None,
    vendor: Optional[str] = None,
    principle_doc_no: Optional[str] = None
):
    """Smart candidate matching for linking document to existing dossier."""
    return await run_in_threadpool(
        find_smart_case_match,
        amount=amount,
        vendor=vendor,
        principle_doc_no=principle_doc_no
    )


@app.get("/api/db/dashboard", tags=["PostgreSQL Database"])
async def api_db_dashboard():
    """Fetch dossier-grouped cases, budget tracking, and document traceability."""
    return await run_in_threadpool(get_dossier_dashboard_data)


@app.post("/api/chatocr/save_to_db", tags=["PostgreSQL Database"])
async def api_chatocr_save_to_db(payload: ChatOcrSaveDbPayload):
    """Save verified/edited document and metadata into PostgreSQL database container with case linking."""
    try:
        res = await run_in_threadpool(
            save_disbursement_document,
            filename=payload.filename or "document.png",
            doc_type=payload.document_type or "general_receipt",
            metadata=payload.metadata or {},
            embed_text=payload.embed_text or "",
            chat_answers=payload.chat_answers or {},
            math_reconciliation=payload.math_reconciliation or {},
            target_record_id=payload.target_record_id,
            is_new_case=payload.is_new_case or False,
            principle_doc_no=payload.principle_doc_no,
            case_title=payload.case_title
        )
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e), "success": False}, status_code=500)


@app.get("/api/db/recent", tags=["PostgreSQL Database"])
async def api_db_recent(limit: int = 10):
    """Fetch recent saved reimbursement documents from PostgreSQL."""
    return await run_in_threadpool(get_recent_saved_documents, limit=limit)



if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Thai Financial Document OCR UI...")
    print("Open your browser at: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)


