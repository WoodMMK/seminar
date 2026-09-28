"""
Component 5: Unified End-to-End Information Extraction Pipeline
Coordinates Ingestion (1), Perception OCR (2), LLM Extraction (3),
and Validation & Business Rules (4) into a cohesive production service.
"""

import base64
import io
import time
from typing import Any, Dict, List, Optional
from PIL import Image

from src.ingestion import DocumentIngestion
from src.ocr_engine import ThaiPerceptionEngine
from src.llm_extractor import LLMExtractor
from src.validator import FinancialDocumentValidator
from src.schemas import (
    DOCUMENT_TYPE_TITLES,
    DocumentType,
    FullPipelineResult,
    HealthCheckResponse,
    PipelineTiming,
)


def _pil_to_base64(img: Image.Image) -> str:
    """Converts PIL image to base64 PNG string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class DocumentProcessingPipeline:
    """
    Unified Pipeline Manager for Thai Financial Document Extraction.
    Sequentially coordinates all 4 underlying components:
      1. Ingestion & Preprocessing (PDF-to-Image, Deskewing)
      2. Thai Perception Engine (PaddleOCR th_PP-OCRv5 & Row grouping)
      3. Reasoning & Extraction Engine (Local LLM via Ollama)
      4. Validation, Normalization & Business Rules Engine
    """

    def __init__(
        self,
        ingestor: Optional[DocumentIngestion] = None,
        ocr_engine: Optional[ThaiPerceptionEngine] = None,
        llm_extractor: Optional[LLMExtractor] = None,
        validator: Optional[FinancialDocumentValidator] = None,
    ):
        self.ingestor = ingestor or DocumentIngestion(target_dpi=150)
        self.ocr_engine = ocr_engine or ThaiPerceptionEngine(
            device="cpu",
            use_doc_unwarping=False,
            use_doc_orientation_classify=False,
            use_textline_orientation=False,
            limit_side_len=1600,
        )
        self.llm_extractor = llm_extractor or LLMExtractor(default_model="qwen2.5:3b")
        self.validator = validator or FinancialDocumentValidator(petty_cash_threshold=10000.0)

    def process(
        self,
        file_bytes: bytes,
        filename: str = "document.png",
        document_type: str = DocumentType.GENERAL_RECEIPT.value,
        model_name: Optional[str] = None,
        auto_deskew: bool = False,
        force_mock: bool = False,
        temperature: float = 0.0,
        include_preview: bool = True,
    ) -> FullPipelineResult:
        """
        Executes the full 4-stage extraction pipeline synchronously.
        """
        total_start = time.perf_counter()

        # ---------------------------------------------------------------------
        # Stage 1: Ingestion & Preprocessing (Component 1)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        pages = self.ingestor.load_document(file_bytes)
        if not pages:
            raise ValueError(f"Could not load or parse document pages from '{filename}'")

        first_page = pages[0]
        image_to_process = first_page.image

        if auto_deskew:
            image_to_process = self.ingestor.deskew_image(image_to_process)

        ingestion_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ---------------------------------------------------------------------
        # Stage 2: Perception Layer (Component 2: PaddleOCR & Row Formatting)
        # ---------------------------------------------------------------------
        t1 = time.perf_counter()
        perception = self.ocr_engine.process_image(image_to_process, page_number=1)
        ocr_ms = round((time.perf_counter() - t1) * 1000, 2)
        ocr_markdown = perception.to_llm_markdown()

        # ---------------------------------------------------------------------
        # Stage 3: LLM Reasoning & Extraction (Component 3)
        # ---------------------------------------------------------------------
        t2 = time.perf_counter()
        extraction_result = self.llm_extractor.extract(
            ocr_markdown=ocr_markdown,
            document_type=document_type,
            model_name=model_name,
            temperature=temperature,
            force_mock=force_mock,
        )
        llm_ms = round((time.perf_counter() - t2) * 1000, 2)

        # ---------------------------------------------------------------------
        # Stage 4: Validation & Normalization Engine (Component 4)
        # ---------------------------------------------------------------------
        t3 = time.perf_counter()
        validation_result = self.validator.validate(extraction_result)
        validation_ms = round((time.perf_counter() - t3) * 1000, 2)

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)

        timing = PipelineTiming(
            ingestion_ms=ingestion_ms,
            ocr_ms=ocr_ms,
            llm_ms=llm_ms,
            validation_ms=validation_ms,
            total_ms=total_ms,
        )

        preview_base64 = _pil_to_base64(image_to_process) if include_preview else None
        type_title = DOCUMENT_TYPE_TITLES.get(document_type, document_type)

        return FullPipelineResult(
            filename=filename,
            document_type=document_type,
            document_type_name_th=type_title,
            pages_count=len(pages),
            timings=timing,
            extraction=extraction_result,
            validation=validation_result,
            raw_ocr_markdown=ocr_markdown,
            total_text_blocks=len(perception.text_blocks),
            image_preview_base64=preview_base64,
        )

    def get_health_status(self) -> HealthCheckResponse:
        """Reports system readiness and status for each pipeline component."""
        ollama_online = self.llm_extractor.is_ollama_running()
        models = self.llm_extractor.list_available_models() if ollama_online else []

        components_status = {
            "component_1_ingestion": {
                "name": "Document Ingestion & Deskew Engine",
                "status": "READY",
                "target_dpi": self.ingestor.target_dpi,
                "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "WEBP", "BMP"],
            },
            "component_2_ocr": {
                "name": "Thai Perception Engine (th_PP-OCRv5)",
                "status": "READY",
                "device": self.ocr_engine.device,
                "limit_side_len": self.ocr_engine.limit_side_len,
            },
            "component_3_llm": {
                "name": "Reasoning & Extraction Engine",
                "status": "ONLINE" if ollama_online else "OFFLINE (Mock Active)",
                "ollama_base_url": self.llm_extractor.base_url,
                "default_model": self.llm_extractor.default_model,
                "installed_models": models,
            },
            "component_4_validator": {
                "name": "Data Validation & Business Rules Engine",
                "status": "READY",
                "supported_document_types": len(DOCUMENT_TYPE_TITLES),
                "petty_cash_threshold_thb": self.validator.petty_cash_threshold,
                "tax_id_algorithm": "Thai Revenue Dept Mod 11 Checksum",
                "date_normalizer": "Thai Buddhist Era (พ.ศ.) & Christian Era (ค.ศ.) -> ISO 8601",
            },
        }

        overall_status = "HEALTHY" if (ollama_online or len(models) >= 0) else "DEGRADED"

        return HealthCheckResponse(
            status=overall_status,
            version="1.0.0",
            components=components_status,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
