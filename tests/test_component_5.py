"""
Comprehensive test suite for Component 5:
Production REST API Service, End-to-End Pipeline & Benchmarking Engine.
Executable directly with: python tests/test_component_5.py
"""

import sys
from pathlib import Path
import io
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.app import app
from src.pipeline import DocumentProcessingPipeline
from src.benchmark import ModelBenchmarker
from src.schemas import (
    DocumentType,
    ValidationSeverity,
    FullPipelineResult,
    BenchmarkReport,
    HealthCheckResponse,
    TypeDirectedExtractionResult
)


def _create_sample_receipt_bytes() -> bytes:
    """Create a high-contrast synthetic receipt image for testing."""
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "ใบเสร็จรับเงิน บริษัท ตัวอย่าง จำกัด", fill=(0, 0, 0))
    draw.text((20, 50), "เลขประจำตัวผู้เสียภาษี 0105558098761", fill=(0, 0, 0))
    draw.text((20, 80), "วันที่ 15 มกราคม 2567", fill=(0, 0, 0))
    draw.text((20, 110), "1. ค่าบริการไอที 1,000.00 บาท", fill=(0, 0, 0))
    draw.text((20, 140), "ยอดรวมทั้งสิ้น 1,000.00 บาท", fill=(0, 0, 0))
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestComponent5(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.sample_bytes = _create_sample_receipt_bytes()

    # =========================================================================
    # 1. Pipeline Unit Tests
    # =========================================================================

    def test_pipeline_health_status(self):
        """Verify DocumentProcessingPipeline health reporting."""
        pipeline = DocumentProcessingPipeline()
        health = pipeline.get_health_status()
        
        self.assertIsInstance(health, HealthCheckResponse)
        self.assertIn(health.status, ["HEALTHY", "DEGRADED"])
        self.assertIn("component_1_ingestion", health.components)
        self.assertIn("component_2_ocr", health.components)
        self.assertIn("component_3_llm", health.components)
        self.assertIn("component_4_validator", health.components)
        self.assertEqual(health.components["component_4_validator"]["status"], "READY")

    @patch("src.llm_extractor.LLMExtractor.extract")
    def test_pipeline_end_to_end_process(self, mock_extract):
        """Verify single-call pipeline execution with stage timing."""
        mock_extract.return_value = TypeDirectedExtractionResult(
            document_type="general_receipt",
            document_type_name_th="ใบเสร็จรับเงินทั่วไป",
            fields={
                "merchant_name": "บริษัท ตัวอย่าง จำกัด",
                "tax_id": "0105558098761",
                "date": "15 มกราคม 2567",
                "subtotal": 1000.0,
                "vat_amount": 0.0,
                "total_amount": 1000.0
            },
            raw_response="{}",
            thinking_process="",
            model_used="qwen2.5:3b",
            latency_seconds=0.1,
            is_mock=False
        )

        pipeline = DocumentProcessingPipeline()
        result = pipeline.process(
            file_bytes=self.sample_bytes,
            filename="test_receipt.png",
            document_type=DocumentType.GENERAL_RECEIPT.value,
        )
        
        self.assertIsInstance(result, FullPipelineResult)
        self.assertEqual(result.filename, "test_receipt.png")
        self.assertEqual(result.document_type, DocumentType.GENERAL_RECEIPT.value)
        self.assertIsNotNone(result.extraction)
        self.assertIsNotNone(result.validation)
        self.assertIsNotNone(result.timings)
        
        # Verify stage timing breakdown
        timing = result.timings
        self.assertGreaterEqual(timing.ocr_ms, 0.0)
        self.assertGreaterEqual(timing.llm_ms, 0.0)
        self.assertGreaterEqual(timing.validation_ms, 0.0)
        self.assertGreaterEqual(timing.total_ms, 0.0)

    # =========================================================================
    # 2. Production REST API Endpoints (/api/v1)
    # =========================================================================

    def test_api_v1_health(self):
        """Test GET /api/v1/health."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["status"], ["HEALTHY", "DEGRADED"])
        self.assertIn("components", data)
        self.assertIn("version", data)

    def test_api_v1_openapi_schema(self):
        """Test OpenAPI schema contains Component 5 paths and tags."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        paths = schema.get("paths", {})
        
        self.assertIn("/api/v1/extract", paths)
        self.assertIn("/api/v1/validate", paths)
        self.assertIn("/api/v1/benchmark", paths)
        self.assertIn("/api/v1/health", paths)

    def test_api_v1_standalone_validate(self):
        """Test POST /api/v1/validate standalone rules engine."""
        payload = {
            "document_type": "general_receipt",
            "document_type_name_th": "ใบเสร็จรับเงินทั่วไป",
            "fields": {
                "vendor_tax_id": "0105558098761",
                "document_date": "15 ม.ค. 2567"
            },
            "line_items": [
                {
                    "item_no": 1,
                    "description": "ค่าอุปกรณ์คอมพิวเตอร์",
                    "quantity": 1.0,
                    "unit_price": 1000.0,
                    "total_price": 1000.0
                }
            ],
            "subtotal": 1000.0,
            "vat": 70.0,
            "total_amount": 1070.0,
            "raw_json": "{}",
            "model_used": "qwen2.5:3b",
            "is_mock": True
        }
        
        response = self.client.post("/api/v1/validate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("status", data)
        self.assertIn("date_report", data)
        self.assertIn("tax_id_report", data)
        self.assertIn("math_report", data)
        self.assertTrue(data["math_report"]["is_balanced"])

    @patch("src.llm_extractor.LLMExtractor.extract")
    def test_api_v1_extract_multipart(self, mock_extract):
        """Test POST /api/v1/extract multipart file upload."""
        mock_extract.return_value = TypeDirectedExtractionResult(
            document_type="general_receipt",
            document_type_name_th="ใบเสร็จรับเงินทั่วไป",
            fields={
                "merchant_name": "บริษัท ตัวอย่าง จำกัด",
                "tax_id": "0105558098761",
                "date": "15 มกราคม 2567",
                "subtotal": 1000.0,
                "vat_amount": 0.0,
                "total_amount": 1000.0
            },
            raw_response="{}",
            thinking_process="",
            model_used="qwen2.5:3b",
            latency_seconds=0.1,
            is_mock=False
        )

        files = {
            "file": ("sample_receipt.png", self.sample_bytes, "image/png")
        }
        data = {
            "document_type": "general_receipt",
            "temperature": "0.0"
        }
        
        response = self.client.post("/api/v1/extract", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        
        self.assertEqual(result["document_type"], "general_receipt")
        self.assertIn("timings", result)
        self.assertIn("extraction", result)
        self.assertIn("validation", result)
        self.assertGreater(result["timings"]["total_ms"], 0)

    def test_api_v1_benchmark(self):
        """Test POST /api/v1/benchmark with snapshot mock baseline."""
        payload = {
            "document_type": "general_receipt",
            "temperature": 0.0,
            "models": ["qwen2.5:3b"]
        }
        
        response = self.client.post("/api/v1/benchmark", json=payload)
        self.assertEqual(response.status_code, 200)
        report = response.json()
        
        self.assertIn("document_type", report)
        self.assertIn("results", report)
        self.assertGreaterEqual(len(report["results"]), 1)
        self.assertIn("markdown_table", report)
        self.assertIn("| Model / Configuration |", report["markdown_table"])

    # =========================================================================
    # 3. Model Benchmarking Engine Direct Tests
    # =========================================================================

    def test_benchmark_engine_direct(self):
        """Verify ModelBenchmarker executes and produces formatted markdown."""
        benchmarker = ModelBenchmarker()
        report = benchmarker.run_benchmark(
            models=["qwen2.5:3b"],
            document_type=DocumentType.GENERAL_RECEIPT.value,
            include_mock_baseline=True
        )
        
        self.assertIsInstance(report, BenchmarkReport)
        self.assertGreaterEqual(len(report.results), 1)
        models = [r.model_name for r in report.results]
        self.assertIn("qwen2.5:3b", models)
        
        # Check Markdown table columns
        table = report.markdown_table
        self.assertIn("| Model / Configuration |", table)
        self.assertIn("| Latency (s) |", table)
        self.assertIn("| Fields Extracted |", table)
        self.assertIn("| Validation Status |", table)
        self.assertIn("| Math Balanced |", table)


if __name__ == "__main__":
    unittest.main()
