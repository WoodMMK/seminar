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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
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
        "name": "PP-ChatOCRv4 (Standard Production)",
        "description": "Primary automated document extraction and Knowledge Payload generator."
    },
    {
        "name": "PostgreSQL & Case Dossier Tracking",
        "description": "Case dossier linking, traceability checklist, and persistence."
    },
    {
        "name": "System & Health",
        "description": "Health checks, connectivity monitoring, and API metadata."
    },
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
    }
]

app = FastAPI(
    title="Thai Financial Document Extraction REST API",
    description=(
        "Senior Project: Web-based Automated Expense Reimbursement System using AI\n\n"
        "Faculty of Engineering, Mahidol University\n\n"
        "Headless REST API powered by PP-ChatOCRv4, Thai PP-OCRv5 Perception, "
        "Local LLM Reasoning (qwen2.5:3b in CPU Mode), and PostgreSQL Case Dossier Tracking."
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# Enable CORS for seamless integration with frontend applications (React, Vue, Next.js, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton instances - unwarping disabled to guarantee pixel-perfect bounding box alignment
# Optimized for CPU execution on standard VMs (150 DPI and 1536px limit)
ingestor = DocumentIngestion(target_dpi=150)
ocr_engine = ThaiPerceptionEngine(
    device="cpu",
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
    limit_side_len=1600
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


def stitch_pages_vertically(pages: list) -> Image.Image:
    """Stitch multiple document pages into a single vertical composite image for unified OCR and multi-page preview."""
    if not pages:
        raise ValueError("No pages to stitch")
    if len(pages) == 1:
        return pages[0].image

    total_width = max(p.image.width for p in pages)
    sep_h = 24
    total_height = sum(p.image.height for p in pages) + (len(pages) - 1) * sep_h
    combined = Image.new("RGB", (total_width, total_height), (226, 232, 240))
    y_offset = 0
    for p in pages:
        x_offset = (total_width - p.image.width) // 2
        combined.paste(p.image, (x_offset, y_offset))
        y_offset += p.image.height + sep_h

    # Cap maximum height to 2400px and width to 1600px to prevent extreme CPU latency on multi-page stitching
    max_composite_h = 2400
    max_composite_w = 1600
    if combined.height > max_composite_h or combined.width > max_composite_w:
        ratio = min(max_composite_h / float(combined.height), max_composite_w / float(combined.width))
        new_w = int(round(combined.width * ratio))
        new_h = int(round(combined.height * ratio))
        combined = combined.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return combined


@app.get("/", tags=["System & Health"], summary="API Root & Service Status")
def api_root():
    return {
        "service": "Thai Financial Document Extraction REST API",
        "version": "1.0.0",
        "architecture": "PP-ChatOCRv4 (PaddleX 3.7.2 + Local LLM CPU Mode)",
        "status": "online",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "chatocr_extraction": "/api/chatocr/chat",
            "chatocr_templates": "/api/chatocr/templates",
            "save_to_database": "/api/chatocr/save_to_db",
            "dossier_cases": "/api/db/cases",
            "db_status": "/api/db/status",
            "recent_documents": "/api/db/recent",
            "health_check": "/api/v1/health"
        }
    }


@app.get("/health", tags=["System & Health"], summary="Quick Liveness Probe")
def health_probe():
    return {"status": "ok"}


from starlette.concurrency import run_in_threadpool

@app.post("/api/preview")
async def preview_document(file: UploadFile = File(...)):
    """Fast preview endpoint converting PDF or image into stitched preview base64."""
    try:
        raw_bytes = await file.read()
        pages = ingestor.load_document(raw_bytes)
        if not pages:
            return JSONResponse({"error": "Failed to parse document pages"}, status_code=400)
        stitched = stitch_pages_vertically(pages)
        return {
            "filename": file.filename,
            "width": stitched.width,
            "height": stitched.height,
            "page_count": len(pages),
            "image_base64": pil_to_base64(stitched)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/extract")
async def extract_document(
    file: UploadFile = File(...),
    auto_deskew: bool = Form(False)
):
    file_bytes = await file.read()
    pages = ingestor.load_document(file_bytes)

    if not pages:
        return JSONResponse({"error": "Failed to parse document pages"}, status_code=400)

    processed_image = stitch_pages_vertically(pages)

    if auto_deskew:
        processed_image = ingestor.deskew_image(processed_image)

    perception = await run_in_threadpool(ocr_engine.process_image, processed_image, page_number=1)

    return {
        "filename": file.filename,
        "width": processed_image.width,
        "height": processed_image.height,
        "page_count": len(pages),
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
                stitched_img = stitch_pages_vertically(pages)
                temp_path = upload_dir / f"pdf_stitched_{int(time.time() * 1000)}.png"
                stitched_img.save(temp_path, format="PNG")
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
    print("Starting Thai Financial Document OCR REST API...")
    print("Swagger Interactive Docs: http://localhost:8000/docs")
    print("ReDoc API Documentation:  http://localhost:8000/redoc")
    print("API Root Status:          http://localhost:8000/")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


